// PyTorch adapter, stage B: device-resident tensors.
//
// torch.classes.scrapgpu.DTensor shares a pooled ScrapGpu buffer and carries a scrapgpu_tensor_desc (ggml
// convention: ne[0] is the fastest dimension). Views (view / reshape / permute / narrow) only change the
// descriptor. Every op allocates a new contiguous result from a size-class buffer pool and enqueues one
// ScrapGpu op on the stream: no host transfer and no stream synchronization happens except in
// from_cpu / to_cpu / sync. An op the backend cannot run raises an error that carries the backend's reason;
// nothing falls back to the CPU.
#include "torch_internal.h"

#include "scrapgpu/attention.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sgtorch {
namespace {

// ---------------------------------------------------------------------------------------------
// buffer pool: NVIDIA 390 probes the tail of every new allocation, so buffers are recycled by size class
// ---------------------------------------------------------------------------------------------

size_t size_class(size_t n) {
    if (n <= 4096) {
        return 4096;
    }
    size_t base = 4096;
    while (base * 2 < n) {
        base *= 2;
    }
    // here base < n <= 2 * base; four classes per octave
    const size_t step = base / 4;
    return base + ((n - base + step - 1) / step) * step;
}

struct Pool {
    std::mutex mu;
    std::map<size_t, std::vector<scrapgpu_buffer_t *>> free_list;
    size_t free_bytes = 0;
    size_t live_bytes = 0;
    size_t max_free = 0;
    std::atomic<uint64_t> creates{0};
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> ops{0};
};

Pool & pool() {
    static Pool * p = [] {
        auto * q = new Pool();
        const char * v = std::getenv("SCRAPGPU_TORCH_POOL_MAX_BYTES");
        q->max_free = v ? static_cast<size_t>(std::strtoull(v, nullptr, 10)) : (256ull << 20);
        return q;
    }();
    return *p;
}

// caller holds pool().mu
void trim_locked(Pool & p) {
    for (auto & kv : p.free_list) {
        for (scrapgpu_buffer_t * b : kv.second) {
            scrapgpu_buffer_free(b);
        }
    }
    p.free_list.clear();
    p.free_bytes = 0;
}

void pool_trim() {
    Pool & p = pool();
    std::lock_guard<std::mutex> lock(p.mu);
    trim_locked(p);
}

void pool_release(scrapgpu_buffer_t * b, size_t cls) {
    Pool & p = pool();
    std::lock_guard<std::mutex> lock(p.mu);
    p.live_bytes -= cls;
    if (p.free_bytes + cls > p.max_free) {
        scrapgpu_buffer_free(b);
        return;
    }
    p.free_list[cls].push_back(b);
    p.free_bytes += cls;
}

struct PBuf {
    scrapgpu_buffer_t * b;
    size_t cls;
    PBuf(scrapgpu_buffer_t * buf, size_t c) : b(buf), cls(c) {}
    PBuf(const PBuf &) = delete;
    PBuf & operator=(const PBuf &) = delete;
    ~PBuf() { pool_release(b, cls); }
};

// caller holds g_mu (g_dev is valid)
std::shared_ptr<PBuf> pool_acquire(size_t bytes) {
    Pool & p = pool();
    const size_t cls = size_class(bytes);
    {
        std::lock_guard<std::mutex> lock(p.mu);
        auto it = p.free_list.find(cls);
        if (it != p.free_list.end() && !it->second.empty()) {
            scrapgpu_buffer_t * b = it->second.back();
            it->second.pop_back();
            p.free_bytes -= cls;
            p.live_bytes += cls;
            p.hits.fetch_add(1, std::memory_order_relaxed);
            return std::make_shared<PBuf>(b, cls);
        }
    }
    scrapgpu_buffer_t * b = nullptr;
    scrapgpu_status st = scrapgpu_buffer_alloc(g_dev, cls, &b);
    if (st != SCRAPGPU_STATUS_OK) {
        // device memory is full of idle pooled buffers: give them back and retry once
        pool_trim();
        b = nullptr;
        st = scrapgpu_buffer_alloc(g_dev, cls, &b);
    }
    if (st != SCRAPGPU_STATUS_OK) {
        size_t live = 0;
        {
            std::lock_guard<std::mutex> lock(p.mu);
            live = p.live_bytes;
        }
        throw std::runtime_error("device tensor allocation of " + std::to_string(cls) + " bytes failed (" +
                                 scrapgpu_status_string(st) + ": " + scrapgpu_last_error() + "); " +
                                 std::to_string(live) + " bytes of DTensors are alive");
    }
    p.creates.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(p.mu);
        p.live_bytes += cls;
    }
    return std::make_shared<PBuf>(b, cls);
}

// ---------------------------------------------------------------------------------------------
// DTensor
// ---------------------------------------------------------------------------------------------

size_t tsize(scrapgpu_dtype dt) {
    return scrapgpu_dtype_type_size(dt);
}

const char * dtype_name(scrapgpu_dtype dt) {
    switch (dt) {
        case SCRAPGPU_DTYPE_F32: return "f32";
        case SCRAPGPU_DTYPE_F16: return "f16";
        case SCRAPGPU_DTYPE_I32: return "i32";
        default: return "other";
    }
}

// the first-fastest strides of a dense tensor
void dense_nb(scrapgpu_tensor_desc & d) {
    d.nb[0] = tsize(d.dtype);
    for (int i = 1; i < 4; ++i) {
        d.nb[i] = d.nb[i - 1] * static_cast<uint64_t>(d.ne[i - 1]);
    }
}

// strides of size-1 dims carry no information: make them the dense value so contiguity is easy to test
void canonicalize(scrapgpu_tensor_desc & d) {
    for (int i = 1; i < 4; ++i) {
        if (d.ne[i] == 1) {
            d.nb[i] = d.nb[i - 1] * static_cast<uint64_t>(d.ne[i - 1]);
        }
    }
}

bool is_dense(const scrapgpu_tensor_desc & d) {
    if (d.nb[0] != tsize(d.dtype)) {
        return false;
    }
    for (int i = 1; i < 4; ++i) {
        if (d.nb[i] != d.nb[i - 1] * static_cast<uint64_t>(d.ne[i - 1])) {
            return false;
        }
    }
    return true;
}

int64_t numel(const scrapgpu_tensor_desc & d) {
    return d.ne[0] * d.ne[1] * d.ne[2] * d.ne[3];
}

// one past the last byte a view touches, relative to the start of its buffer
uint64_t extent(const scrapgpu_tensor_desc & d) {
    uint64_t end = d.offset + tsize(d.dtype);
    for (int i = 0; i < 4; ++i) {
        end += static_cast<uint64_t>(d.ne[i] - 1) * d.nb[i];
    }
    return end;
}

class DTensorImpl : public torch::CustomClassHolder {
public:
    DTensorImpl(std::shared_ptr<PBuf> buf, const scrapgpu_tensor_desc & desc) : buf_(std::move(buf)), d_(desc) {
        d_.buffer = buf_->b;
    }
    const scrapgpu_tensor_desc & desc() const { return d_; }
    const std::shared_ptr<PBuf> & buf() const { return buf_; }

private:
    std::shared_ptr<PBuf> buf_;
    scrapgpu_tensor_desc d_;
};

using DT = c10::intrusive_ptr<DTensorImpl>;

// caller holds g_mu: a new dense tensor in a pooled buffer
DT alloc_dense(scrapgpu_dtype dt, int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
    TORCH_CHECK(n0 > 0 && n1 > 0 && n2 > 0 && n3 > 0, "DTensor dimensions must be positive");
    scrapgpu_tensor_desc d = make_desc(dt, SCRAPGPU_LAYOUT_NATURAL, n0, n1, n2, n3, nullptr);
    const size_t bytes = static_cast<size_t>(numel(d)) * tsize(dt);
    return c10::make_intrusive<DTensorImpl>(pool_acquire(bytes), d);
}

DT make_view(const DT & src, const scrapgpu_tensor_desc & d) {
    return c10::make_intrusive<DTensorImpl>(src->buf(), d);
}

bool g_sync_each_op = false;

// caller holds g_mu. Checks the backend can run the op (reporting its reason) and enqueues it.
void run(scrapgpu_op op, const char * name, const std::vector<scrapgpu_tensor_desc> & ins,
         std::vector<scrapgpu_tensor_desc> & outs, const void * params, size_t params_size) {
    scrapgpu_op_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.op = op;
    desc.inputs = ins.data();
    desc.input_count = ins.size();
    desc.outputs = outs.data();
    desc.output_count = outs.size();
    desc.params = params;
    desc.params_size = params_size;
    check_status(scrapgpu_op_desc_validate(&desc), name);
    const char * reason = nullptr;
    const scrapgpu_status sup = scrapgpu_op_supported(g_dev, &desc, &reason);
    if (sup != SCRAPGPU_STATUS_OK) {
        throw std::runtime_error(std::string("ScrapGpu ") + name + " is not supported by the " + g_backend +
                                 " backend for these tensors (no CPU fallback): " +
                                 (reason ? reason : scrapgpu_last_error()));
    }
    check_status(scrapgpu_op_execute(g_stream, &desc), name);
    pool().ops.fetch_add(1, std::memory_order_relaxed);
    if (g_sync_each_op) {
        check_status(scrapgpu_stream_sync(g_stream), name);
    }
}

void begin() {
    ensure_backend();
    static bool once = [] {
        const char * v = std::getenv("SCRAPGPU_TORCH_SYNC_EACH_OP");
        g_sync_each_op = v != nullptr && v[0] != '\0' && v[0] != '0';
        return true;
    }();
    (void) once;
}

scrapgpu_dtype dtype_of(const at::Tensor & t) {
    switch (t.scalar_type()) {
        case at::kFloat: return SCRAPGPU_DTYPE_F32;
        case at::kInt: return SCRAPGPU_DTYPE_I32;
        case at::kHalf: return SCRAPGPU_DTYPE_F16;
        default: TORCH_CHECK(false, "DTensor supports float32, float16 and int32 CPU tensors");
    }
    return SCRAPGPU_DTYPE_UNKNOWN;
}

at::ScalarType torch_dtype(scrapgpu_dtype dt) {
    switch (dt) {
        case SCRAPGPU_DTYPE_F32: return at::kFloat;
        case SCRAPGPU_DTYPE_I32: return at::kInt;
        case SCRAPGPU_DTYPE_F16: return at::kHalf;
        default: TORCH_CHECK(false, "unsupported DTensor dtype");
    }
    return at::kFloat;
}

// ---- host <-> device (the only places the adapter moves data) ----

DT dt_from_cpu(const at::Tensor & t) {
    TORCH_CHECK(t.device().is_cpu(), "from_cpu expects a CPU tensor");
    TORCH_CHECK(t.dim() >= 1 && t.dim() <= 4, "from_cpu expects a 1-D to 4-D tensor");
    TORCH_CHECK(t.numel() > 0, "from_cpu expects a non-empty tensor");
    TORCH_CHECK(!t.requires_grad(), "ScrapGpu supports inference only");
    const scrapgpu_dtype dt = dtype_of(t);
    const at::Tensor c = t.contiguous();
    int64_t ne[4] = {1, 1, 1, 1};
    for (int64_t i = 0; i < c.dim(); ++i) {
        ne[i] = c.size(c.dim() - 1 - i);  // torch's last dim is the fastest one
    }
    std::lock_guard<std::mutex> lock(g_mu);
    begin();
    DT out = alloc_dense(dt, ne[0], ne[1], ne[2], ne[3]);
    const size_t bytes = static_cast<size_t>(c.numel()) * tsize(dt);
    check_status(scrapgpu_tensor_upload(g_stream, &out->desc(), c.const_data_ptr(), bytes), "DTensor upload");
    return out;
}

// caller holds g_mu
DT cont_locked(const DT & x) {
    if (is_dense(x->desc())) {
        return x;
    }
    const scrapgpu_tensor_desc & s = x->desc();
    TORCH_CHECK(s.dtype == SCRAPGPU_DTYPE_F32 || s.dtype == SCRAPGPU_DTYPE_F16, "cont supports f32 and f16 tensors");
    DT out = alloc_dense(s.dtype, s.ne[0], s.ne[1], s.ne[2], s.ne[3]);
    std::vector<scrapgpu_tensor_desc> outs = {out->desc()};
    run(SCRAPGPU_OP_CONT, "CONT", {s}, outs, nullptr, 0);
    return out;
}

at::Tensor dt_to_cpu(const DT & x) {
    std::lock_guard<std::mutex> lock(g_mu);
    begin();
    const DT c = cont_locked(x);
    const scrapgpu_tensor_desc & d = c->desc();
    at::Tensor out = at::empty({d.ne[3], d.ne[2], d.ne[1], d.ne[0]}, at::TensorOptions().dtype(torch_dtype(d.dtype)));
    const size_t bytes = static_cast<size_t>(numel(d)) * tsize(d.dtype);
    check_status(scrapgpu_tensor_download(g_stream, &d, out.data_ptr(), bytes), "DTensor download");
    return out;
}

void dt_sync() {
    std::lock_guard<std::mutex> lock(g_mu);
    begin();
    check_status(scrapgpu_stream_sync(g_stream), "stream_sync");
}

// ---- views (descriptor only) ----

std::vector<int64_t> dt_ne(const DT & x) {
    return {x->desc().ne[0], x->desc().ne[1], x->desc().ne[2], x->desc().ne[3]};
}
std::vector<int64_t> dt_nb(const DT & x) {
    return {static_cast<int64_t>(x->desc().nb[0]), static_cast<int64_t>(x->desc().nb[1]),
            static_cast<int64_t>(x->desc().nb[2]), static_cast<int64_t>(x->desc().nb[3])};
}

DT dt_view(const DT & x, const std::vector<int64_t> & ne, const std::vector<int64_t> & nb, int64_t offset) {
    TORCH_CHECK(ne.size() == 4 && nb.size() == 4, "view expects 4 sizes and 4 byte strides");
    TORCH_CHECK(offset >= 0, "view offset must be nonnegative");
    scrapgpu_tensor_desc d = x->desc();
    for (int i = 0; i < 4; ++i) {
        TORCH_CHECK(ne[i] > 0 && nb[i] > 0, "view sizes and strides must be positive");
        d.ne[i] = ne[i];
        d.nb[i] = static_cast<uint64_t>(nb[i]);
    }
    d.offset = static_cast<uint64_t>(offset);
    TORCH_CHECK(extent(d) <= x->buf()->cls, "view exceeds the tensor's buffer");
    canonicalize(d);
    return make_view(x, d);
}

DT dt_reshape(const DT & x, const std::vector<int64_t> & ne) {
    TORCH_CHECK(ne.size() == 4, "reshape expects 4 sizes");
    TORCH_CHECK(is_dense(x->desc()), "reshape needs a contiguous tensor; call cont() first");
    int64_t n = 1;
    scrapgpu_tensor_desc d = x->desc();
    for (int i = 0; i < 4; ++i) {
        TORCH_CHECK(ne[i] > 0, "reshape sizes must be positive");
        n *= ne[i];
        d.ne[i] = ne[i];
    }
    TORCH_CHECK(n == numel(x->desc()), "reshape must keep the number of elements");
    dense_nb(d);
    return make_view(x, d);
}

// result.dim i = x.dim order[i]
DT dt_permute(const DT & x, const std::vector<int64_t> & order) {
    TORCH_CHECK(order.size() == 4, "permute expects 4 axes");
    bool seen[4] = {false, false, false, false};
    scrapgpu_tensor_desc d = x->desc();
    for (int i = 0; i < 4; ++i) {
        TORCH_CHECK(order[i] >= 0 && order[i] < 4 && !seen[order[i]], "permute expects a permutation of 0..3");
        seen[order[i]] = true;
        d.ne[i] = x->desc().ne[order[i]];
        d.nb[i] = x->desc().nb[order[i]];
    }
    canonicalize(d);
    return make_view(x, d);
}

DT dt_narrow(const DT & x, int64_t dim, int64_t start, int64_t length) {
    TORCH_CHECK(dim >= 0 && dim < 4, "narrow dim must be 0..3");
    TORCH_CHECK(start >= 0 && length > 0 && start + length <= x->desc().ne[dim], "narrow range is outside the tensor");
    scrapgpu_tensor_desc d = x->desc();
    d.offset += static_cast<uint64_t>(start) * d.nb[dim];
    d.ne[dim] = length;
    canonicalize(d);
    return make_view(x, d);
}

DT dt_cont(const DT & x) {
    std::lock_guard<std::mutex> lock(g_mu);
    begin();
    return cont_locked(x);
}

// ---- ops: every result is a new dense DTensor ----

void require_f32(const DT & x, const char * what) {
    TORCH_CHECK(x->desc().dtype == SCRAPGPU_DTYPE_F32, what, " must be float32");
}

DT dt_mul_mat(const c10::intrusive_ptr<WeightImpl> & w, const DT & x) {
    std::lock_guard<std::mutex> lock(g_mu);
    begin();
    require_f32(x, "mul_mat input");
    const scrapgpu_tensor_desc & xd = x->desc();
    TORCH_CHECK(xd.ne[0] == w->in_features(), "mul_mat: x.ne[0] (", xd.ne[0], ") must equal in_features (", w->in_features(), ")");
    DT out = alloc_dense(SCRAPGPU_DTYPE_F32, w->out_features(), xd.ne[1], xd.ne[2], xd.ne[3]);
    scrapgpu_mul_mat_params mm{};
    mm.struct_size = sizeof(mm);
    mm.allow_broadcast = 1;
    std::vector<scrapgpu_tensor_desc> outs = {out->desc()};
    run(SCRAPGPU_OP_MUL_MAT, "MUL_MAT", {w->desc(), xd}, outs, &mm, sizeof(mm));
    return out;
}

// An uninitialized dense tensor (a destination for mul_mat_into). Contents are undefined until written.
DT dt_empty(const std::vector<int64_t> & ne, const std::string & dtype) {
    TORCH_CHECK(ne.size() == 4, "empty expects 4 sizes");
    TORCH_CHECK(dtype == "f32" || dtype == "f16" || dtype == "i32", "empty dtype must be f32, f16 or i32");
    std::lock_guard<std::mutex> lock(g_mu);
    begin();
    return alloc_dense(dtype == "f32" ? SCRAPGPU_DTYPE_F32 : dtype == "f16" ? SCRAPGPU_DTYPE_F16 : SCRAPGPU_DTYPE_I32,
                       ne[0], ne[1], ne[2], ne[3]);
}

// dst = w @ x written into an existing F32 tensor or view of shape [out, x.ne1, x.ne2, x.ne3] (for example a column
// block of a larger result). The only op that writes into an existing DTensor: the caller owns dst's contents.
DT dt_mul_mat_into(const c10::intrusive_ptr<WeightImpl> & w, const DT & x, const DT & dst) {
    std::lock_guard<std::mutex> lock(g_mu);
    begin();
    require_f32(x, "mul_mat input");
    require_f32(dst, "mul_mat destination");
    const scrapgpu_tensor_desc & xd = x->desc();
    const scrapgpu_tensor_desc & dd = dst->desc();
    TORCH_CHECK(xd.ne[0] == w->in_features(), "mul_mat: x.ne[0] (", xd.ne[0], ") must equal in_features (", w->in_features(), ")");
    TORCH_CHECK(dd.ne[0] == w->out_features() && dd.ne[1] == xd.ne[1] && dd.ne[2] == xd.ne[2] && dd.ne[3] == xd.ne[3],
                "mul_mat_into: destination must be [out_features, x.ne[1], x.ne[2], x.ne[3]]");
    scrapgpu_mul_mat_params mm{};
    mm.struct_size = sizeof(mm);
    mm.allow_broadcast = 1;
    std::vector<scrapgpu_tensor_desc> outs = {dd};
    run(SCRAPGPU_OP_MUL_MAT, "MUL_MAT", {w->desc(), xd}, outs, &mm, sizeof(mm));
    return dst;
}

DT binary(scrapgpu_op op, const char * name, const DT & a, const DT & b) {
    std::lock_guard<std::mutex> lock(g_mu);
    begin();
    require_f32(a, "lhs");
    require_f32(b, "rhs");
    const scrapgpu_tensor_desc & ad = a->desc();
    const scrapgpu_tensor_desc & bd = b->desc();
    for (int i = 0; i < 4; ++i) {
        TORCH_CHECK(bd.ne[i] <= ad.ne[i] && ad.ne[i] % bd.ne[i] == 0,
                    name, ": rhs must broadcast to lhs (rhs.ne[", i, "]=", bd.ne[i], ", lhs.ne[", i, "]=", ad.ne[i], ")");
    }
    DT out = alloc_dense(SCRAPGPU_DTYPE_F32, ad.ne[0], ad.ne[1], ad.ne[2], ad.ne[3]);
    std::vector<scrapgpu_tensor_desc> outs = {out->desc()};
    run(op, name, {ad, bd}, outs, nullptr, 0);
    return out;
}

DT dt_add(const DT & a, const DT & b) { return binary(SCRAPGPU_OP_ADD, "ADD", a, b); }
DT dt_mul(const DT & a, const DT & b) { return binary(SCRAPGPU_OP_MUL, "MUL", a, b); }

DT dt_unary(const DT & x, int64_t kind) {
    TORCH_CHECK(kind >= SCRAPGPU_UNARY_SILU && kind <= SCRAPGPU_UNARY_TANH, "unknown unary kind");
    std::lock_guard<std::mutex> lock(g_mu);
    begin();
    require_f32(x, "unary input");
    const scrapgpu_tensor_desc & xd = x->desc();
    DT out = alloc_dense(SCRAPGPU_DTYPE_F32, xd.ne[0], xd.ne[1], xd.ne[2], xd.ne[3]);
    scrapgpu_unary_params p{};
    p.struct_size = sizeof(p);
    p.kind = static_cast<scrapgpu_unary_kind>(kind);
    std::vector<scrapgpu_tensor_desc> outs = {out->desc()};
    run(SCRAPGPU_OP_UNARY, "UNARY", {xd}, outs, &p, sizeof(p));
    return out;
}

// out = act(first half) * second half when swapped == 0; act(second half) * first half when swapped == 1
DT dt_glu(const DT & x, int64_t kind, int64_t swapped) {
    TORCH_CHECK(kind == SCRAPGPU_GLU_SWIGLU || kind == SCRAPGPU_GLU_REGLU, "unknown GLU kind");
    std::lock_guard<std::mutex> lock(g_mu);
    begin();
    require_f32(x, "glu input");
    const scrapgpu_tensor_desc & xd = x->desc();
    TORCH_CHECK(xd.ne[0] % 2 == 0, "glu: ne[0] must be even");
    DT out = alloc_dense(SCRAPGPU_DTYPE_F32, xd.ne[0] / 2, xd.ne[1], xd.ne[2], xd.ne[3]);
    scrapgpu_glu_params p{};
    p.struct_size = sizeof(p);
    p.kind = static_cast<scrapgpu_glu_kind>(kind);
    p.split_inputs = 0;
    p.swapped = static_cast<int>(swapped);
    std::vector<scrapgpu_tensor_desc> outs = {out->desc()};
    run(SCRAPGPU_OP_GLU, "GLU", {xd}, outs, &p, sizeof(p));
    return out;
}

DT dt_rms_norm_mul(const DT & x, const DT & w, double eps) {
    std::lock_guard<std::mutex> lock(g_mu);
    begin();
    require_f32(x, "rms_norm input");
    require_f32(w, "rms_norm weight");
    const scrapgpu_tensor_desc & xd = x->desc();
    TORCH_CHECK(w->desc().ne[0] == xd.ne[0] && numel(w->desc()) == xd.ne[0], "rms_norm weight must be a single row of x.ne[0] floats");
    DT out = alloc_dense(SCRAPGPU_DTYPE_F32, xd.ne[0], xd.ne[1], xd.ne[2], xd.ne[3]);
    scrapgpu_rms_norm_params p{};
    p.struct_size = sizeof(p);
    p.eps = static_cast<float>(eps);
    std::vector<scrapgpu_tensor_desc> outs = {out->desc()};
    run(SCRAPGPU_OP_RMS_NORM_MUL, "RMS_NORM_MUL", {xd, w->desc()}, outs, &p, sizeof(p));
    return out;
}

// x: [head_dim, heads, tokens, batch]; pos: int32 [tokens] shared by every batch element
DT dt_rope(const DT & x, const DT & pos, int64_t n_dims, int64_t n_ctx_orig, double freq_base, double freq_scale,
           double ext_factor, double attn_factor, double beta_fast, double beta_slow, int64_t mode) {
    std::lock_guard<std::mutex> lock(g_mu);
    begin();
    require_f32(x, "rope input");
    TORCH_CHECK(pos->desc().dtype == SCRAPGPU_DTYPE_I32, "rope positions must be int32");
    const scrapgpu_tensor_desc & xd = x->desc();
    DT out = alloc_dense(SCRAPGPU_DTYPE_F32, xd.ne[0], xd.ne[1], xd.ne[2], xd.ne[3]);
    scrapgpu_rope_params rp{};
    rp.struct_size = sizeof(rp);
    rp.mode = mode ? SCRAPGPU_ROPE_MODE_NEOX : SCRAPGPU_ROPE_MODE_NORMAL;
    rp.n_dims = static_cast<int32_t>(n_dims > 0 ? n_dims : xd.ne[0]);
    rp.n_ctx_orig = static_cast<int32_t>(n_ctx_orig);
    rp.freq_base = static_cast<float>(freq_base);
    rp.freq_scale = static_cast<float>(freq_scale);
    rp.ext_factor = static_cast<float>(ext_factor);
    rp.attn_factor = static_cast<float>(attn_factor);
    rp.beta_fast = static_cast<float>(beta_fast);
    rp.beta_slow = static_cast<float>(beta_slow);
    std::vector<scrapgpu_tensor_desc> outs = {out->desc()};
    run(SCRAPGPU_OP_ROPE, "ROPE", {xd, pos->desc()}, outs, &rp, sizeof(rp));
    return out;
}

DT dt_concat(const DT & a, const DT & b, int64_t dim) {
    TORCH_CHECK(dim >= 0 && dim < 4, "concat dim must be 0..3");
    std::lock_guard<std::mutex> lock(g_mu);
    begin();
    const scrapgpu_tensor_desc & ad = a->desc();
    const scrapgpu_tensor_desc & bd = b->desc();
    TORCH_CHECK(ad.dtype == bd.dtype, "concat needs matching dtypes");
    int64_t ne[4];
    for (int i = 0; i < 4; ++i) {
        TORCH_CHECK(i == dim || ad.ne[i] == bd.ne[i], "concat: dimensions other than dim must match");
        ne[i] = ad.ne[i] + (i == dim ? bd.ne[i] : 0);
    }
    DT out = alloc_dense(ad.dtype, ne[0], ne[1], ne[2], ne[3]);
    scrapgpu_concat_params p{};
    p.struct_size = sizeof(p);
    p.dim = static_cast<uint32_t>(dim);
    std::vector<scrapgpu_tensor_desc> outs = {out->desc()};
    run(SCRAPGPU_OP_CONCAT, "CONCAT", {ad, bd}, outs, &p, sizeof(p));
    return out;
}

// out = x * s + bias
DT dt_scale(const DT & x, double s, double bias) {
    std::lock_guard<std::mutex> lock(g_mu);
    begin();
    require_f32(x, "scale input");
    const scrapgpu_tensor_desc & xd = x->desc();
    DT out = alloc_dense(SCRAPGPU_DTYPE_F32, xd.ne[0], xd.ne[1], xd.ne[2], xd.ne[3]);
    scrapgpu_scale_params p{};
    p.struct_size = sizeof(p);
    p.scale = static_cast<float>(s);
    p.bias = static_cast<float>(bias);
    std::vector<scrapgpu_tensor_desc> outs = {out->desc()};
    run(SCRAPGPU_OP_SCALE, "SCALE", {xd}, outs, &p, sizeof(p));
    return out;
}

// q=[D,Q,Hq,B], k=[D,K,Hkv,B], v=[Dv,K,Hkv,B] F32 (non-contiguous inputs are made contiguous on the device);
// mask: additive F32 [K,Q,1|Hq,1|B] (-inf blocks a key); a query whose keys are all blocked yields zeros
DT sdpa(const DT & q_in, const DT & k_in, const DT & v_in, const DT * mask_in, double scale) {
    std::lock_guard<std::mutex> lock(g_mu);
    begin();
    require_f32(q_in, "query");
    require_f32(k_in, "key");
    require_f32(v_in, "value");
    const DT q = cont_locked(q_in), k = cont_locked(k_in), v = cont_locked(v_in);
    DT mask;
    if (mask_in) {
        require_f32(*mask_in, "mask");
        mask = cont_locked(*mask_in);
    }
    const scrapgpu_tensor_desc qd = q->desc(), kd = k->desc(), vd = v->desc();
    DT out = alloc_dense(SCRAPGPU_DTYPE_F32, vd.ne[0], qd.ne[1], qd.ne[2], qd.ne[3]);
    scrapgpu_tensor_desc od = out->desc();
    scrapgpu_tensor_desc md{};
    scrapgpu_attention_params p{};
    p.struct_size = sizeof(p);
    p.query = &qd;
    p.key = &kd;
    p.value = &vd;
    if (mask.defined()) {
        md = mask->desc();
        p.mask = &md;
    }
    p.output = &od;
    p.scale = static_cast<float>(scale);
    p.flags = SCRAPGPU_ATTENTION_FLAG_DEVICE_BATCHED;
    uint64_t need = 0;
    check_status(scrapgpu_attention_batched_workspace_bytes(&p, &need), "attention workspace size");
    // the workspace goes back to the pool when this call returns; the stream is in order, so later ops that
    // reuse it run after the attention kernels
    const std::shared_ptr<PBuf> ws = pool_acquire(static_cast<size_t>(need));
    p.workspace = ws->b;
    p.workspace_bytes = ws->cls;
    check_status(scrapgpu_scaled_dot_product_attention(g_dev, g_stream, &p), "scaled_dot_product_attention (device batched)");
    pool().ops.fetch_add(1, std::memory_order_relaxed);
    if (g_sync_each_op) {
        check_status(scrapgpu_stream_sync(g_stream), "sdpa");
    }
    return out;
}

DT dt_sdpa(const DT & q, const DT & k, const DT & v, double scale) {
    return sdpa(q, k, v, nullptr, scale);
}

DT dt_sdpa_mask(const DT & q, const DT & k, const DT & v, const DT & mask, double scale) {
    return sdpa(q, k, v, &mask, scale);
}

// [host_to_device_calls, host_to_device_bytes, device_to_host_calls, device_to_host_bytes,
//  ops, pool_buffer_creates, pool_hits, pool_idle_bytes, dtensor_live_bytes]
std::vector<int64_t> dt_stats() {
    scrapgpu_transfer_stats ts{};
    ts.struct_size = sizeof(ts);
    check_status(scrapgpu_transfer_stats_get(&ts), "transfer stats");
    Pool & p = pool();
    std::lock_guard<std::mutex> lock(p.mu);
    return {static_cast<int64_t>(ts.host_to_device_calls), static_cast<int64_t>(ts.host_to_device_bytes),
            static_cast<int64_t>(ts.device_to_host_calls), static_cast<int64_t>(ts.device_to_host_bytes),
            static_cast<int64_t>(p.ops.load()), static_cast<int64_t>(p.creates.load()),
            static_cast<int64_t>(p.hits.load()), static_cast<int64_t>(p.free_bytes), static_cast<int64_t>(p.live_bytes)};
}

void dt_stats_reset() {
    scrapgpu_transfer_stats_reset();
    Pool & p = pool();
    p.ops = 0;
    p.creates = 0;
    p.hits = 0;
}

void dt_pool_trim() {
    pool_trim();
}

} // namespace

void register_device_tensor(torch::Library & m) {
    m.class_<DTensorImpl>("DTensor")
        .def("ne", &dt_ne)
        .def("nb", &dt_nb)
        .def("offset", [](const DT & x) { return static_cast<int64_t>(x->desc().offset); })
        .def("dtype", [](const DT & x) { return std::string(dtype_name(x->desc().dtype)); })
        .def("is_contiguous", [](const DT & x) { return is_dense(x->desc()); })
        .def("to_cpu", &dt_to_cpu)
        .def("view", &dt_view)
        .def("reshape", &dt_reshape)
        .def("permute", &dt_permute)
        .def("narrow", &dt_narrow)
        .def("cont", &dt_cont);
    m.def("dt_from_cpu", &dt_from_cpu);
    m.def("dt_sync", &dt_sync);
    m.def("dt_mul_mat", &dt_mul_mat);
    m.def("dt_empty", &dt_empty);
    m.def("dt_mul_mat_into", &dt_mul_mat_into);
    m.def("dt_add", &dt_add);
    m.def("dt_mul", &dt_mul);
    m.def("dt_unary", &dt_unary);
    m.def("dt_glu", &dt_glu);
    m.def("dt_rms_norm_mul", &dt_rms_norm_mul);
    m.def("dt_rope", &dt_rope);
    m.def("dt_concat", &dt_concat);
    m.def("dt_scale", &dt_scale);
    m.def("dt_sdpa", &dt_sdpa);
    m.def("dt_sdpa_mask", &dt_sdpa_mask);
    m.def("dt_stats", &dt_stats);
    m.def("dt_stats_reset", &dt_stats_reset);
    m.def("dt_pool_trim", &dt_pool_trim);
}

} // namespace sgtorch
