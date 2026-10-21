// SGLX composite device (see composite.h): the ggml device, buffer type and buffer, the row split of a weight over the physical devices
// (pdev.h) and the execution of MUL_MAT / GET_ROWS.
//
// Buffer: plain host memory (the host shadow). Every tensor of the buffer is fully there, get_tensor / set_tensor are memcpy, so the sched
// copies of activations and results around this device are cheap and a weight can be (re)distributed from it. A weight that a MUL_MAT uses
// is registered on its first use (split_rec): its rows are cut into contiguous ranges, one per physical device in proportion to the split
// ratios (aligned to 32 rows so that the repacked layout of CL11 works on whole groups), each range is created and uploaded on its
// device. A set_tensor / memset / clear / copy that touches the bytes of a registered weight marks it for a new upload.
//
// MUL_MAT: the columns of src1 go to every device first (submit() returns once the work is enqueued), then the devices are waited for
// one after the other, so they run at the same time. N = 1 (token generation): every device writes its rows directly into the result.
//
// Not supported, so that ggml's scheduler runs them elsewhere: MUL_MAT with batch dimensions or a permuted weight, types that not every
// device of the split can compute, and every other op. GET_ROWS (the input embedding of a model that shares the tensor with its output
// projection) is answered from the host shadow with the dequantizing functions of ggml.

#include "composite.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <malloc.h>
#else
#  include <dlfcn.h>
#endif

namespace sglx {

double default_weight_for(const std::string & description, bool is_cpu) {
    // effective GB/s of a quantized mat-vec: docs/plans/MOE_COMPOSITE_DEVICE.md 3.2 and docs/benchmarks/moe-ep-20261021/c1-profile
    static const struct { const char * key; double gbs; } table[] = {
        { "rx 6400", 116.0 }, { "wx 2100", 41.0 }, { "gt 730", 25.0 }, { "gt 710", 7.6 }, { "gt 430", 15.0 },
    };
    std::string d = description;
    std::transform(d.begin(), d.end(), d.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    for (const auto & e : table) {
        if (d.find(e.key) != std::string::npos) {
            return e.gbs;
        }
    }
    return is_cpu ? 10.0 : 20.0;
}

namespace {

using clock_type = std::chrono::steady_clock;

double ms_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

// ---- profile ----

struct acc {
    double   sum = 0.0;
    double   mx  = 0.0;
    uint64_t n   = 0;
    void add(double v) { sum += v; mx = std::max(mx, v); ++n; }
    double mean() const { return n != 0 ? sum / (double) n : 0.0; }
};

struct prof_set {
    acc total;                   // whole MUL_MAT (gather .. scatter)
    acc gather;
    acc scatter;
    std::vector<acc> submit;     // per physical device: time inside submit()
    std::vector<acc> done;       // per physical device: time since the start of the call at which its wait() returned
};

// ---- the split of one weight ----

struct part {
    int      pdev = 0;
    int64_t  r0   = 0;
    int64_t  r1   = 0;
    piece *  p    = nullptr;
};

struct split_rec {
    char *    data = nullptr;
    size_t    nbytes = 0;
    ggml_type type = GGML_TYPE_F32;
    int64_t   K = 0;
    int64_t   M = 0;
    size_t    row_bytes = 0;
    bool      uploaded = false;
    bool      failed = false;
    std::vector<part> parts;
};

struct buffer_ctx {
    char *                 base = nullptr;
    size_t                 size = 0;
    std::vector<split_rec> recs;
};

struct composite {
    ggml_backend_reg_t reg = nullptr;
    std::vector<std::pair<std::string, cl11_device *>> cl11;
    std::vector<std::pair<std::string, double>> spec; // device name, ratio (< 0: default)

    std::mutex mtx;
    bool inited = false;
    bool ok = false;
    std::vector<std::unique_ptr<pdev>> pdevs;
    std::vector<double> weights;      // normalized, same order as pdevs
    std::map<int, bool> type_ok;

    std::string desc;
    ggml_backend_buffer_type buft = {};
    ggml_backend_device      dev  = {};

    bool profile = false;
    bool serial = false;              // GGML_SGLX_SERIAL=1: wait for each device right after its submit (to measure what the overlap gives)
    int  profile_every = 128;
    prof_set prof[2];                 // [0]: N == 1, [1]: N > 1
    acc upload_ms;
    std::vector<float> xs;            // gathered input columns
    std::vector<std::vector<float>> outs; // result staging per part (N > 1)
};

composite * g_c = nullptr; // never destroyed: the physical devices (ggml backends) must not be torn down during static destruction

bool buffer_owned(ggml_backend_buffer_t buffer);

// ---- log ----

#define SGLX_LOG(...) fprintf(stderr, "SGLX: " __VA_ARGS__)

// ---- initialization of the physical devices ----

// The registry functions (ggml_backend_dev_by_name) are in libggml, which loads this plugin and cannot be linked back: ask the process.
ggml_backend_dev_t find_ggml_device(const char * name) {
#if defined(_WIN32)
    (void) name;
    return nullptr;
#else
    using fn_t = ggml_backend_dev_t (*)(const char *);
    static fn_t fn = (fn_t) dlsym(RTLD_DEFAULT, "ggml_backend_dev_by_name");
    return fn != nullptr ? fn(name) : nullptr;
#endif
}

bool ensure_init(composite * c) {
    std::lock_guard<std::mutex> lock(c->mtx);
    if (c->inited) {
        return c->ok;
    }
    c->inited = true;
    c->ok = false;

    size_t n_ratio = 0;
    for (const auto & s : c->spec) {
        n_ratio += s.second >= 0.0 ? 1 : 0;
    }
    if (n_ratio != 0 && n_ratio != c->spec.size()) {
        SGLX_LOG("GGML_SGLX_SPLIT: give a ratio to every device or to none\n");
        return false;
    }
    for (const auto & s : c->spec) {
        std::unique_ptr<pdev> p;
        if (s.first.rfind("CL11_", 0) == 0) {
            for (const auto & d : c->cl11) {
                if (d.first == s.first) {
                    p = make_pdev_cl11(d.second);
                }
            }
            if (!p) {
                SGLX_LOG("GGML_SGLX_SPLIT: no CL11 device %s\n", s.first.c_str());
                return false;
            }
        } else {
            ggml_backend_dev_t d = find_ggml_device(s.first.c_str());
            if (d == nullptr) {
                SGLX_LOG("GGML_SGLX_SPLIT: no ggml device %s (is its backend loaded?)\n", s.first.c_str());
                return false;
            }
            p = make_pdev_ggml(d);
            if (!p) {
                return false;
            }
        }
        c->pdevs.push_back(std::move(p));
    }
    double sum = 0.0;
    for (size_t i = 0; i < c->pdevs.size(); ++i) {
        const double w = n_ratio != 0 ? c->spec[i].second : c->pdevs[i]->default_weight();
        c->weights.push_back(w);
        sum += w;
    }
    if (sum <= 0.0) {
        SGLX_LOG("GGML_SGLX_SPLIT: the ratios add up to zero\n");
        return false;
    }
    std::string line;
    for (size_t i = 0; i < c->pdevs.size(); ++i) {
        c->weights[i] /= sum;
        char buf[160];
        snprintf(buf, sizeof(buf), "%s%s (%s) %.3f", i != 0 ? ", " : "", c->pdevs[i]->name().c_str(), c->pdevs[i]->description().c_str(), c->weights[i]);
        line += buf;
    }
    SGLX_LOG("SGLX0 splits the rows of its weights over: %s\n", line.c_str());
    c->prof[0].submit.resize(c->pdevs.size());
    c->prof[0].done.resize(c->pdevs.size());
    c->prof[1].submit.resize(c->pdevs.size());
    c->prof[1].done.resize(c->pdevs.size());
    c->ok = true;
    return true;
}

// a type that every device of the split can compute
bool mul_mat_type_ok(composite * c, ggml_type type) {
    if (!ensure_init(c)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(c->mtx);
    auto it = c->type_ok.find((int) type);
    if (it != c->type_ok.end()) {
        return it->second;
    }
    bool ok = true;
    for (auto & p : c->pdevs) {
        ok = ok && p->supports_mul_mat(type);
    }
    c->type_ok[(int) type] = ok;
    return ok;
}

// ---- profile output ----

void print_profile(composite * c) {
    if (!c->profile) {
        return;
    }
    for (int k = 0; k < 2; ++k) {
        const prof_set & ps = c->prof[k];
        if (ps.total.n == 0) {
            continue;
        }
        std::string per;
        for (size_t j = 0; j < c->pdevs.size(); ++j) {
            char buf[200];
            snprintf(buf, sizeof(buf), "  %s: submit %.3f ms, done at %.3f ms (max %.3f)", c->pdevs[j]->name().c_str(), ps.submit[j].mean(), ps.done[j].mean(), ps.done[j].mx);
            per += buf;
        }
        fprintf(stderr, "SGLX profile MUL_MAT %s: calls %llu, mean %.3f ms (max %.3f), gather %.3f, scatter %.3f;%s\n", k == 0 ? "N=1" : "N>1",
                (unsigned long long) ps.total.n, ps.total.mean(), ps.total.mx, ps.gather.mean(), ps.scatter.mean(), per.c_str());
    }
    if (c->upload_ms.n != 0) {
        fprintf(stderr, "SGLX profile upload: %llu weights, %.1f ms in total\n", (unsigned long long) c->upload_ms.n, c->upload_ms.sum);
    }
}

void print_profile_atexit() {
    if (g_c != nullptr) {
        print_profile(g_c);
    }
}

// ---- the split ----

// contiguous row ranges in proportion to the weights, the boundaries multiples of 32 (the last range takes the rest)
std::vector<part> make_plan(const composite * c, int64_t M) {
    std::vector<part> plan;
    int last = -1;
    for (size_t j = 0; j < c->pdevs.size(); ++j) {
        if (c->weights[j] > 0.0) {
            last = (int) j;
        }
    }
    int64_t prev = 0;
    double cum = 0.0;
    for (size_t j = 0; j < c->pdevs.size(); ++j) {
        if (c->weights[j] <= 0.0) {
            continue;
        }
        cum += c->weights[j];
        int64_t b = (int) j == last ? M : (int64_t) std::llround((double) M * cum / 32.0) * 32;
        b = std::min(std::max(b, prev), M);
        if (b > prev) {
            part p;
            p.pdev = (int) j;
            p.r0 = prev;
            p.r1 = b;
            plan.push_back(p);
            prev = b;
        }
    }
    return plan;
}

void free_rec(composite * c, split_rec & r) {
    for (auto & p : r.parts) {
        if (p.p != nullptr) {
            c->pdevs[p.pdev]->piece_free(p.p);
            p.p = nullptr;
        }
    }
    r.parts.clear();
    r.uploaded = false;
}

// the record of the weight `w` in its buffer (created on first use); nullptr if the devices cannot hold it
split_rec * get_rec(composite * c, buffer_ctx * bc, const ggml_tensor * w) {
    char * data = (char *) w->data;
    split_rec * rec = nullptr;
    for (auto & r : bc->recs) {
        if (r.data == data) {
            rec = &r;
        }
    }
    const size_t row_bytes = ggml_row_size(w->type, w->ne[0]);
    if (rec != nullptr && (rec->type != w->type || rec->K != w->ne[0] || rec->M != w->ne[1])) {
        free_rec(c, *rec); // the same bytes used as another matrix
        rec->failed = false;
    }
    if (rec == nullptr) {
        bc->recs.emplace_back();
        rec = &bc->recs.back();
    }
    rec->data = data;
    rec->type = w->type;
    rec->K = w->ne[0];
    rec->M = w->ne[1];
    rec->row_bytes = row_bytes;
    rec->nbytes = row_bytes * (size_t) w->ne[1];
    if (rec->failed) {
        return nullptr;
    }
    if (rec->parts.empty()) {
        rec->parts = make_plan(c, rec->M);
        for (auto & p : rec->parts) {
            p.p = c->pdevs[p.pdev]->piece_create(rec->type, rec->K, p.r1 - p.r0);
            if (p.p == nullptr) {
                SGLX_LOG("%s cannot hold %lld rows of %s (%.1f MiB); change GGML_SGLX_SPLIT\n", c->pdevs[p.pdev]->name().c_str(), (long long) (p.r1 - p.r0),
                         ggml_type_name(rec->type), (double) (row_bytes * (size_t) (p.r1 - p.r0)) / 1048576.0);
                free_rec(c, *rec);
                rec->failed = true;
                return nullptr;
            }
        }
    }
    if (!rec->uploaded) {
        const auto t0 = clock_type::now();
        for (auto & p : rec->parts) {
            if (!c->pdevs[p.pdev]->piece_upload(p.p, rec->data + (size_t) p.r0 * row_bytes, (size_t) (p.r1 - p.r0) * row_bytes)) {
                SGLX_LOG("upload of %lld rows to %s failed\n", (long long) (p.r1 - p.r0), c->pdevs[p.pdev]->name().c_str());
                free_rec(c, *rec);
                rec->failed = true;
                return nullptr;
            }
        }
        rec->uploaded = true;
        const double ms = ms_since(t0);
        c->upload_ms.add(ms);
        if (c->profile) {
            std::string s;
            for (auto & p : rec->parts) {
                char buf[100];
                snprintf(buf, sizeof(buf), " %s[%lld..%lld)", c->pdevs[p.pdev]->name().c_str(), (long long) p.r0, (long long) p.r1);
                s += buf;
            }
            SGLX_LOG("uploaded %.1f MiB (%s %lld x %lld) in %.1f ms:%s\n", (double) rec->nbytes / 1048576.0, ggml_type_name(rec->type), (long long) rec->K, (long long) rec->M, ms, s.c_str());
        }
    }
    return rec;
}

// ---- buffer ----

buffer_ctx * bctx_of(ggml_backend_buffer_t buffer) {
    return (buffer_ctx *) buffer->context;
}

void note_write(buffer_ctx * bc, const char * p, size_t n) {
    for (auto & r : bc->recs) {
        if (r.data < p + n && p < r.data + r.nbytes) {
            r.uploaded = false;
        }
    }
}

const char * buft_get_name(ggml_backend_buffer_type_t) {
    return "SGLX0";
}

void buffer_free(ggml_backend_buffer_t buffer) {
    buffer_ctx * bc = bctx_of(buffer);
    if (g_c != nullptr) {
        for (auto & r : bc->recs) {
            free_rec(g_c, r);
        }
    }
#if defined(_WIN32)
    _aligned_free(bc->base);
#else
    free(bc->base);
#endif
    delete bc;
}

void * buffer_get_base(ggml_backend_buffer_t buffer) {
    return bctx_of(buffer)->base;
}

void buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
    note_write(bctx_of(buffer), (char *) tensor->data + offset, size);
}

void buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
    note_write(bctx_of(buffer), (char *) tensor->data + offset, size);
}

void buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
}

bool buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t sb = src->view_src != nullptr ? src->view_src->buffer : src->buffer;
    if (sb == nullptr || !(ggml_backend_buffer_is_host(sb) || buffer_owned(sb))) {
        return false;
    }
    const size_t n = ggml_nbytes(src);
    memmove(dst->data, src->data, n);
    note_write(bctx_of(buffer), (const char *) dst->data, n);
    return true;
}

void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    buffer_ctx * bc = bctx_of(buffer);
    memset(bc->base, value, bc->size);
    for (auto & r : bc->recs) {
        r.uploaded = false;
    }
}

const ggml_backend_buffer_i buffer_i = {
    /* .free_buffer   = */ buffer_free,
    /* .get_base      = */ buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ buffer_memset_tensor,
    /* .set_tensor    = */ buffer_set_tensor,
    /* .get_tensor    = */ buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ buffer_cpy_tensor,
    /* .clear         = */ buffer_clear,
    /* .reset         = */ nullptr,
};

bool buffer_owned(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->iface.free_buffer == buffer_free;
}

constexpr size_t kAlignment = 64;

ggml_backend_buffer_t buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t alloc = std::max<size_t>(kAlignment, (size + kAlignment - 1) & ~(kAlignment - 1));
    void * base = nullptr;
#if defined(_WIN32)
    base = _aligned_malloc(alloc, 4096);
#else
    if (posix_memalign(&base, 4096, alloc) != 0) {
        base = nullptr;
    }
#endif
    if (base == nullptr) {
        SGLX_LOG("cannot allocate %.1f MiB of host memory\n", (double) size / 1048576.0);
        return nullptr;
    }
    auto * bc = new buffer_ctx();
    bc->base = (char *) base;
    bc->size = alloc;
    return ggml_backend_buffer_init(buft, buffer_i, bc, alloc);
}

size_t buft_get_alignment(ggml_backend_buffer_type_t) {
    return kAlignment;
}

bool buft_is_host(ggml_backend_buffer_type_t) {
    // false on purpose: with true the loader reads straight into the tensors and bypasses set_tensor, which tells the buffer when a weight changed
    return false;
}

const ggml_backend_buffer_type_i buft_i = {
    /* .get_name         = */ buft_get_name,
    /* .alloc_buffer     = */ buft_alloc_buffer,
#if GGML_BACKEND_API_VERSION >= 3
    /* .alloc_buffer_n   = */ nullptr,
#endif
    /* .get_alignment    = */ buft_get_alignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ nullptr,
#if GGML_BACKEND_API_VERSION >= 3
    /* .get_alloc_size_n = */ nullptr,
#endif
    /* .is_host          = */ buft_is_host,
};

// ---- ops ----

bool mul_mat_shape_ok(composite * c, const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (w == nullptr || x == nullptr) {
        return false;
    }
    if (w->buffer != nullptr || w->view_src != nullptr) {
        ggml_backend_buffer_t wb = w->view_src != nullptr ? w->view_src->buffer : w->buffer;
        if (wb != nullptr && !buffer_owned(wb)) {
            return false; // a weight of another device
        }
    }
    if (w->ne[2] != 1 || w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1 || op->ne[2] != 1 || op->ne[3] != 1) {
        return false; // no batch dimensions
    }
    if (x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || w->ne[0] != x->ne[0] || op->ne[0] != w->ne[1] || op->ne[1] != x->ne[1]) {
        return false;
    }
    if (w->ne[0] % ggml_blck_size(w->type) != 0) {
        return false;
    }
    if (w->nb[0] != ggml_type_size(w->type) || w->nb[1] != ggml_row_size(w->type, w->ne[0])) {
        return false; // a permuted or padded weight
    }
    if (x->nb[0] != sizeof(float) || op->nb[0] != sizeof(float) || op->nb[1] != (size_t) op->ne[0] * sizeof(float)) {
        return false;
    }
    return mul_mat_type_ok(c, w->type);
}

bool get_rows_shape_ok(const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * ids = op->src[1];
    if (w == nullptr || ids == nullptr || ids->type != GGML_TYPE_I32) {
        return false;
    }
    ggml_backend_buffer_t wb = w->view_src != nullptr ? w->view_src->buffer : w->buffer;
    if (wb != nullptr && !buffer_owned(wb)) {
        return false;
    }
    if (w->ne[2] != ids->ne[1] || w->ne[3] != ids->ne[2] || op->ne[0] != w->ne[0] || op->ne[1] != ids->ne[0]) {
        return false;
    }
    if (w->nb[0] != ggml_type_size(w->type) || w->ne[0] % ggml_blck_size(w->type) != 0) {
        return false;
    }
    if (w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_I32) {
        return op->type == w->type && op->nb[0] == ggml_type_size(w->type);
    }
    const ggml_type_traits * tr = ggml_get_type_traits(w->type);
    return tr != nullptr && tr->to_float != nullptr && op->type == GGML_TYPE_F32 && op->nb[0] == sizeof(float);
}

bool do_get_rows(const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * ids = op->src[1];
    const int64_t nc = w->ne[0];
    const ggml_type_traits * tr = ggml_get_type_traits(w->type);
    for (int64_t i12 = 0; i12 < ids->ne[2]; ++i12) {
        for (int64_t i11 = 0; i11 < ids->ne[1]; ++i11) {
            for (int64_t i10 = 0; i10 < ids->ne[0]; ++i10) {
                const int64_t i01 = *(const int32_t *) ((const char *) ids->data + i10 * ids->nb[0] + i11 * ids->nb[1] + i12 * ids->nb[2]);
                if (i01 < 0 || i01 >= w->ne[1]) {
                    SGLX_LOG("GET_ROWS: index %lld out of range (%lld rows)\n", (long long) i01, (long long) w->ne[1]);
                    return false;
                }
                const char * s = (const char *) w->data + i01 * w->nb[1] + i11 * w->nb[2] + i12 * w->nb[3];
                char * d = (char *) op->data + i10 * op->nb[1] + i11 * op->nb[2] + i12 * op->nb[3];
                if (w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_I32) {
                    memcpy(d, s, (size_t) nc * ggml_type_size(w->type));
                } else {
                    tr->to_float(s, (float *) d, nc);
                }
            }
        }
    }
    return true;
}

bool do_mul_mat(composite * c, ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    ggml_backend_buffer_t wb = w->view_src != nullptr ? w->view_src->buffer : w->buffer;
    if (!buffer_owned(wb)) {
        SGLX_LOG("MUL_MAT: the weight is not in an SGLX0 buffer\n");
        return false;
    }
    const auto t0 = clock_type::now();
    split_rec * rec = get_rec(c, bctx_of(wb), w);
    if (rec == nullptr) {
        return false;
    }
    const int64_t K = w->ne[0];
    const int64_t M = w->ne[1];
    const int64_t N = x->ne[1];
    prof_set & ps = c->prof[N == 1 ? 0 : 1];

    // the input columns: used in place when they are contiguous, gathered otherwise
    const float * xp = (const float *) x->data;
    if (N > 1 && x->nb[1] != (size_t) K * sizeof(float)) {
        c->xs.resize((size_t) K * (size_t) N);
        for (int64_t n = 0; n < N; ++n) {
            memcpy(c->xs.data() + (size_t) n * (size_t) K, (const char *) x->data + (size_t) n * x->nb[1], (size_t) K * sizeof(float));
        }
        xp = c->xs.data();
    }
    float * dst = (float *) op->data;
    if (c->outs.size() < rec->parts.size()) {
        c->outs.resize(rec->parts.size());
    }
    std::vector<float *> outp(rec->parts.size());
    for (size_t i = 0; i < rec->parts.size(); ++i) {
        const part & p = rec->parts[i];
        if (N == 1) {
            outp[i] = dst + p.r0;
        } else {
            c->outs[i].resize((size_t) (p.r1 - p.r0) * (size_t) N);
            outp[i] = c->outs[i].data();
        }
    }
    const double t_gather = ms_since(t0);

    // start every device, then wait for them (a device is waited for once, after its last piece)
    std::vector<double> t_submit(c->pdevs.size(), 0.0);
    std::vector<double> t_done(c->pdevs.size(), -1.0);
    for (size_t i = 0; i < rec->parts.size(); ++i) {
        const part & p = rec->parts[i];
        const auto ts = clock_type::now();
        if (!c->pdevs[p.pdev]->submit(p.p, xp, N, outp[i])) {
            // drain what was started: the staging buffers must not be released under a running device
            for (auto & q : c->pdevs) {
                q->wait();
            }
            return false;
        }
        t_submit[p.pdev] += ms_since(ts);
        if (c->serial) {
            c->pdevs[p.pdev]->wait();
            t_done[p.pdev] = ms_since(t0);
        }
    }
    bool ok = true;
    for (size_t j = 0; j < c->pdevs.size(); ++j) {
        bool used = false;
        for (const auto & p : rec->parts) {
            used = used || p.pdev == (int) j;
        }
        if (used && !c->serial) {
            ok = c->pdevs[j]->wait() && ok;
            t_done[j] = ms_since(t0);
        }
    }
    if (!ok) {
        return false;
    }

    const auto tsc = clock_type::now();
    if (N > 1) {
        for (size_t i = 0; i < rec->parts.size(); ++i) {
            const part & p = rec->parts[i];
            const int64_t rows = p.r1 - p.r0;
            for (int64_t n = 0; n < N; ++n) {
                memcpy(dst + (size_t) n * (size_t) M + p.r0, outp[i] + (size_t) n * (size_t) rows, (size_t) rows * sizeof(float));
            }
        }
    }
    if (c->profile) {
        ps.total.add(ms_since(t0));
        ps.gather.add(t_gather);
        ps.scatter.add(ms_since(tsc));
        for (size_t j = 0; j < c->pdevs.size(); ++j) {
            if (t_done[j] >= 0.0) {
                ps.submit[j].add(t_submit[j]);
                ps.done[j].add(t_done[j]);
            }
        }
        if (N == 1 && c->profile_every > 0 && ps.total.n % (uint64_t) c->profile_every == 0) {
            print_profile(c);
        }
    }
    return true;
}

// ---- backend ----

struct backend_ctx {
    composite * c;
};

const char * backend_get_name(ggml_backend_t) {
    return "SGLX0";
}

void backend_free(ggml_backend_t backend) {
    delete (backend_ctx *) backend->context;
    delete backend;
}

void backend_synchronize(ggml_backend_t) {
    // every op is finished when graph_compute returns
}

enum ggml_status backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    composite * c = ((backend_ctx *) backend->context)->c;
    if (!ensure_init(c)) {
        return GGML_STATUS_FAILED;
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_is_empty(node) || ggml_op_is_empty(node->op)) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_GET_ROWS:
                if (!do_get_rows(node)) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_MUL_MAT:
                if (!do_mul_mat(c, node)) {
                    return GGML_STATUS_FAILED;
                }
                break;
            default:
                SGLX_LOG("op %s is not supported\n", ggml_op_desc(node));
                return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

const ggml_backend_i backend_i = {
    /* .get_name                = */ backend_get_name,
    /* .free                    = */ backend_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ backend_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ backend_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

ggml_guid_t backend_guid() {
    static ggml_guid guid = { 0x53, 0x47, 0x4c, 0x58, 0x63, 0x6f, 0x6d, 0x70, 0x6f, 0x73, 0x69, 0x74, 0x65, 0x64, 0x65, 0x76 };
    return &guid;
}

// ---- device ----

composite * dev_c(ggml_backend_dev_t dev) {
    return (composite *) dev->context;
}

const char * dev_get_name(ggml_backend_dev_t) {
    return "SGLX0";
}

const char * dev_get_description(ggml_backend_dev_t dev) {
    return dev_c(dev)->desc.c_str();
}

void dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    // the weights live in the pinned-down memory of the physical devices and in host memory; nothing here limits the placement
    *total = (size_t) 8 << 30;
    *free  = *total;
}

enum ggml_backend_dev_type dev_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

void dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = dev_get_name(dev);
    props->description = dev_get_description(dev);
    props->type = dev_get_type(dev);
    props->device_id = nullptr;
    dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

ggml_backend_t dev_init_backend(ggml_backend_dev_t dev, const char *) {
    composite * c = dev_c(dev);
    if (!ensure_init(c)) {
        return nullptr;
    }
    return new ggml_backend {
        /* .guid    = */ backend_guid(),
        /* .iface   = */ backend_i,
        /* .device  = */ dev,
        /* .context = */ new backend_ctx{ c },
    };
}

ggml_backend_buffer_type_t dev_get_buffer_type(ggml_backend_dev_t dev) {
    return &dev_c(dev)->buft;
}

bool dev_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    composite * c = dev_c(dev);
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT:
            return mul_mat_shape_ok(c, op);
        case GGML_OP_GET_ROWS:
            return get_rows_shape_ok(op);
        default:
            return false;
    }
}

bool dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == buft_get_name && buft->context == dev->context;
}

const ggml_backend_device_i dev_i = {
    /* .get_name             = */ dev_get_name,
    /* .get_description      = */ dev_get_description,
    /* .get_memory           = */ dev_get_memory,
    /* .get_type             = */ dev_get_type,
    /* .get_props            = */ dev_get_props,
    /* .init_backend         = */ dev_init_backend,
    /* .get_buffer_type      = */ dev_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ dev_supports_op,
    /* .supports_buft        = */ dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

} // namespace

// ---- registration ----

void attach(ggml_backend_reg_t reg, const std::vector<std::pair<std::string, cl11_device *>> & cl11) {
    const char * env = getenv("GGML_SGLX_SPLIT");
    if (env == nullptr || *env == '\0' || g_c != nullptr) {
        return;
    }
    auto * c = new composite();
    c->reg = reg;
    c->cl11 = cl11;
    std::string s(env);
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t end = s.find(',', pos);
        if (end == std::string::npos) {
            end = s.size();
        }
        std::string item = s.substr(pos, end - pos);
        pos = end + 1;
        if (item.empty()) {
            continue;
        }
        double ratio = -1.0;
        const size_t colon = item.find(':');
        if (colon != std::string::npos) {
            ratio = atof(item.c_str() + colon + 1);
            item.resize(colon);
        }
        c->spec.emplace_back(item, ratio);
    }
    if (c->spec.empty()) {
        delete c;
        return;
    }
    c->profile = getenv("GGML_SGLX_PROFILE") != nullptr && atoi(getenv("GGML_SGLX_PROFILE")) != 0;
    c->serial = getenv("GGML_SGLX_SERIAL") != nullptr && atoi(getenv("GGML_SGLX_SERIAL")) != 0;
    if (getenv("GGML_SGLX_PROFILE_EVERY") != nullptr) {
        c->profile_every = atoi(getenv("GGML_SGLX_PROFILE_EVERY"));
    }
    c->desc = "SGLX composite (" + s + ")";
    c->buft = ggml_backend_buffer_type{ /* .iface = */ buft_i, /* .device = */ &c->dev, /* .context = */ c };
    c->dev = ggml_backend_device{ /* .iface = */ dev_i, /* .reg = */ reg, /* .context = */ c };
    g_c = c;
    atexit(print_profile_atexit);
}

size_t device_count() {
    return g_c != nullptr ? 1 : 0;
}

ggml_backend_dev_t device_get(size_t index) {
    GGML_ASSERT(g_c != nullptr && index == 0);
    return &g_c->dev;
}

} // namespace sglx
