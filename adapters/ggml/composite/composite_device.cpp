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
#include "moe_block.h"
#include "sgl_trace.h"

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
#  include <sys/mman.h>
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
    uint64_t  gen = 0;                // composite::gen of the plan in `parts` (a newer one: cut the rows again with the tuned ratios)
    std::vector<part> parts;
};

struct buffer_ctx {
    char *                 base = nullptr;
    size_t                 size = 0;
    bool                   external = false;   // wraps memory of someone else (the mmap of a model file, pinned memory): the memory itself is not freed
    ggml_backend_buffer_t  pinned = nullptr;   // MoE mode: the pinned host buffer (of the first MoE device) that provides the memory; freed with this buffer
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
    bool threaded_last = false;       // GGML_SGLX_THREADED_LAST=1: the pdevs with a worker thread are submitted after the others
    bool lazy_copy = true;            // GGML_SGLX_LAZY_COPY=0 turns the lazy input copy (cpy_tensor_async) off
    bool hostbuf = false;             // GGML_SGLX_HOSTBUF=1
    bool trace = false;               // GGML_SGLX_TRACE=1 (with GGML_SGLX_PROFILE): one line per graph_compute call
    bool serial = false;              // GGML_SGLX_SERIAL=1: wait for each device right after its submit (to measure what the overlap gives)
    int  profile_every = 128;
    uint64_t prof_calls = 0;
    // phases of one token seen from this device: GET_ROWS (input embedding) -> [the layers on the other devices] -> MUL_MAT -> [logits, sampling, next graph]
    clock_type::time_point t_gr_end{}, t_mm_end{};
    acc gap_layers, gap_host, gr_ms;
    acc gr_big_ms, gc_big_ms;          // GET_ROWS with more than one index / whole graph_compute calls, any N (GGML_SGLX_PROFILE)
    // GGML_SGLX_AUTOTUNE=<rounds>[,<calls>]: measure every device alone (serial submit + wait) for <calls> token-generation MUL_MATs (default 24),
    // set the ratios so that all devices finish together (time alone = a + b * rows, b fitted from two rounds), and re-plan the weights; <rounds> times
    int      at_rounds = 0;
    int      at_calls = 24;
    int      at_round = 0;             // rounds finished
    int      at_seen = 0;              // N == 1 calls since the last re-plan
    std::vector<double> at_time, at_rows; // per pdev, sums over the measured calls
    std::vector<std::vector<std::pair<double, double>>> at_hist; // per pdev: (rows, ms alone) of every finished round
    uint64_t gen = 0;
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
ggml_backend_dev_t find_ggml_device_impl(const char * name) {
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

    if (c->spec.empty()) {
        c->ok = true; // MoE only (GGML_SGLX_MOE): no physical device for the rows of a weight
        return true;
    }
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
        } else if (s.first.size() > 4 && s.first.rfind("CUDA", 0) == 0 && s.first.find_first_not_of("0123456789", 4) == std::string::npos) {
            p = make_pdev_cuda(atoi(s.first.c_str() + 4));
            if (!p) {
                SGLX_LOG("GGML_SGLX_SPLIT: no CUDA device %s (plugin built without CL11_WITH_CUDA, no libcuda, or not in SCRAPGPU_CUDA_DEVICES)\n", s.first.c_str());
                return false;
            }
        } else if (s.first.size() > 6 && s.first.rfind("REMOTE", 0) == 0 && s.first.find_first_not_of("0123456789", 6) == std::string::npos) {
            p = make_pdev_scrapgpu(atoi(s.first.c_str() + 6));
            if (!p) {
                SGLX_LOG("GGML_SGLX_SPLIT: no remote device %s (plugin built without CL11_WITH_REMOTE, or the scrapgpu-device-server could not be started: see GGML_SGLX_REMOTE_*)\n", s.first.c_str());
                return false;
            }
        } else {
            ggml_backend_dev_t d = find_ggml_device_impl(s.first.c_str());
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
            const std::string st = c->pdevs[j]->stats();
            if (!st.empty()) {
                per += " [" + st + "]";
            }
        }
        fprintf(stderr, "SGLX profile MUL_MAT %s: calls %llu, mean %.3f ms (max %.3f), gather %.3f, scatter %.3f;%s\n", k == 0 ? "N=1" : "N>1",
                (unsigned long long) ps.total.n, ps.total.mean(), ps.total.mx, ps.gather.mean(), ps.scatter.mean(), per.c_str());
    }
    if (c->gap_layers.n != 0) {
        fprintf(stderr, "SGLX profile token phases (N=1): GET_ROWS %.3f ms; layers on the other devices (GET_ROWS end -> MUL_MAT start) %.3f ms; after MUL_MAT (-> next GET_ROWS: logits, sampling, next graph) %.3f ms\n",
                c->gr_ms.mean(), c->gap_layers.mean(), c->gap_host.mean());
    }
    if (c->gr_big_ms.n != 0) {
        fprintf(stderr, "SGLX profile GET_ROWS with N>1: %llu calls, mean %.3f ms (max %.3f); graph_compute calls (any N): %llu, mean %.3f ms (max %.3f)\n",
                (unsigned long long) c->gr_big_ms.n, c->gr_big_ms.mean(), c->gr_big_ms.mx, (unsigned long long) c->gc_big_ms.n, c->gc_big_ms.mean(), c->gc_big_ms.mx);
    }
    if (c->upload_ms.n != 0) {
        fprintf(stderr, "SGLX profile upload: %llu weights, %.1f ms in total\n", (unsigned long long) c->upload_ms.n, c->upload_ms.sum);
    }
}

void print_profile_atexit() {
    if (g_c != nullptr) {
        print_profile(g_c);
        moe::print_profile();
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
    if (rec != nullptr && rec->gen != c->gen && !rec->parts.empty()) {
        free_rec(c, *rec); // the ratios were tuned: cut the rows again
    }
    if (rec == nullptr) {
        bc->recs.emplace_back();
        rec = &bc->recs.back();
    }
    rec->gen = c->gen;
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
    if (bc->pinned != nullptr) {
        ggml_backend_buffer_free(bc->pinned);
    } else if (!bc->external) {
#if defined(_WIN32)
        _aligned_free(bc->base);
#else
        free(bc->base);
#endif
    }
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

// a tensor placed in the buffer: the expert tensors of a MoE model are registered for moe_block.cpp
enum ggml_status buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor * tensor) {
    if (moe::enabled()) {
        moe::note_tensor(tensor);
    }
    return GGML_STATUS_SUCCESS;
}

const ggml_backend_buffer_i buffer_i = {
    /* .free_buffer   = */ buffer_free,
    /* .get_base      = */ buffer_get_base,
    /* .init_tensor   = */ buffer_init_tensor,
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
    if (moe::enabled() && size > 0) {
        // MoE mode: the activations that the scheduler copies in and out of this device (and the compute buffers) live in pinned memory of the first GPU, so that the copies can be
        // asynchronous: one submission for the three inputs of a layer instead of three blocking reads
        ggml_backend_buffer_t pb = moe::alloc_pinned(alloc);
        if (pb != nullptr) {
            auto * bc = new buffer_ctx();
            bc->base = (char *) ggml_backend_buffer_get_base(pb);
            bc->size = alloc;
            bc->external = true;
            bc->pinned = pb;
            return ggml_backend_buffer_init(buft, buffer_i, bc, alloc);
        }
    }
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
    // false by default, on purpose: with true the loader reads straight into the tensors and bypasses set_tensor, which tells the buffer when a weight changed.
    // GGML_SGLX_HOSTBUF=1: true, and GET_ROWS is left to the CPU backend (which can then read the shared embedding in place); the weights are read when the
    // first MUL_MAT needs them, after the load, and a model weight does not change afterwards.
    return g_c != nullptr && g_c->hostbuf;
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
    if (w == nullptr || x == nullptr || c->spec.empty()) {
        return false; // no GGML_SGLX_SPLIT: SGLX0 only runs MoE ops
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

bool get_rows_shape_ok(const composite * c, const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * ids = op->src[1];
    if (w == nullptr || ids == nullptr || ids->type != GGML_TYPE_I32 || c->spec.empty()) {
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

// one measured call of GGML_SGLX_AUTOTUNE; after the last one of a round the ratios are set and the weights are cut again on their next use
void autotune_note(composite * c, const std::vector<double> & t_alone, const std::vector<double> & n_rows) {
    if (c->at_time.size() != c->pdevs.size()) {
        c->at_time.assign(c->pdevs.size(), 0.0);
        c->at_rows.assign(c->pdevs.size(), 0.0);
    }
    for (size_t j = 0; j < c->pdevs.size(); ++j) {
        c->at_time[j] += t_alone[j];
        c->at_rows[j] += n_rows[j];
    }
    if (c->at_seen < 3 + c->at_calls) {
        return;
    }
    const size_t n = c->pdevs.size();
    const double calls = (double) c->at_calls;
    std::string line;
    double total_rows = 0.0;
    if (c->at_hist.size() != n) {
        c->at_hist.assign(n, {});
    }
    for (size_t j = 0; j < n; ++j) {
        const double rows = c->at_rows[j] / calls;
        const double ms = c->at_time[j] / calls;
        total_rows += rows;
        if (rows > 0.0) {
            c->at_hist[j].emplace_back(rows, ms);
        }
        char buf[120];
        snprintf(buf, sizeof(buf), "%s%s %.3f ms (%.0f rows)", j != 0 ? ", " : "", c->pdevs[j]->name().c_str(), ms, rows);
        line += buf;
    }
    ++c->at_round;
    // time of a device alone = a + b * rows (a: transfers and launches, b: the kernel): b from the last two rounds when the rows differ
    // enough, else a = 0 (proportional); then every device gets the rows that make them all finish together: a_j + b_j * r_j = T, sum r_j = M
    std::vector<double> a(n, 0.0), b(n, 0.0), r(n, 0.0);
    std::vector<bool> live(n, false);
    for (size_t j = 0; j < n; ++j) {
        const auto & h = c->at_hist[j];
        if (c->weights[j] <= 0.0 || h.empty()) {
            continue;
        }
        live[j] = true;
        const auto last = h.back();
        b[j] = last.second / last.first;
        if (h.size() >= 2) {
            const auto prev = h[h.size() - 2];
            const double dr = last.first - prev.first;
            const double dt = last.second - prev.second;
            if (std::fabs(dr) > 0.03 * last.first && dt / dr > 0.02 * b[j]) {
                b[j] = dt / dr;
                a[j] = std::max(0.0, last.second - b[j] * last.first);
            }
        }
    }
    double T = 0.0;
    for (int it = 0; it < (int) n; ++it) {
        double inv = 0.0, ab = 0.0;
        for (size_t j = 0; j < n; ++j) {
            if (live[j]) {
                inv += 1.0 / b[j];
                ab += a[j] / b[j];
            }
        }
        if (inv <= 0.0) {
            break;
        }
        T = (total_rows + ab) / inv;
        bool dropped = false;
        for (size_t j = 0; j < n; ++j) {
            if (live[j] && T <= a[j]) { // slower than T before it computes anything: leave it out
                live[j] = false;
                dropped = true;
            }
        }
        if (!dropped) {
            break;
        }
    }
    double wsum = 0.0;
    for (size_t j = 0; j < n; ++j) {
        r[j] = live[j] ? std::max(0.0, (T - a[j]) / b[j]) : 0.0;
        // a step of at most a factor 2 per round
        if (r[j] > 0.0 && !c->at_hist[j].empty()) {
            const double cur = c->at_hist[j].back().first;
            r[j] = std::min(std::max(r[j], 0.5 * cur), 2.0 * cur);
        }
        wsum += r[j];
    }
    if (wsum > 0.0) {
        std::string nw;
        for (size_t j = 0; j < n; ++j) {
            c->weights[j] = r[j] / wsum;
            char buf[80];
            snprintf(buf, sizeof(buf), "%s%s %.3f", j != 0 ? ", " : "", c->pdevs[j]->name().c_str(), c->weights[j]);
            nw += buf;
        }
        SGLX_LOG("autotune round %d: alone %s -> ratios %s (predicted %.2f ms)\n", c->at_round, line.c_str(), nw.c_str(), T);
        ++c->gen;
    }
    c->at_seen = 0;
    std::fill(c->at_time.begin(), c->at_time.end(), 0.0);
    std::fill(c->at_rows.begin(), c->at_rows.end(), 0.0);
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
    // the pdevs with a worker thread first: their work starts while the host thread submits the others
    std::vector<size_t> order;
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < rec->parts.size(); ++i) {
            if (c->pdevs[rec->parts[i].pdev]->threaded() == ((pass == 0) == !c->threaded_last)) {
                order.push_back(i);
            }
        }
    }
    // GGML_SGLX_AUTOTUNE: a call that measures every device alone
    const bool measuring = c->at_round < c->at_rounds && N == 1 && c->at_seen >= 3 && c->at_seen < 3 + c->at_calls;
    if (N == 1 && c->at_round < c->at_rounds) {
        ++c->at_seen;
    }
    std::vector<double> t_alone(c->pdevs.size(), 0.0);
    std::vector<double> n_rows(c->pdevs.size(), 0.0);
    if (measuring) {
        // the pdevs with a worker thread are submitted in the same order but waited for at once: serial for all
        order.clear();
        for (size_t i = 0; i < rec->parts.size(); ++i) {
            order.push_back(i);
        }
    }
    if (N == 1) {
        SGLX_TRACE("mm_b");
    }
    for (size_t i : order) {
        const part & p = rec->parts[i];
        const auto ts = clock_type::now();
        if (N == 1) {
            SGLX_TRACE("sub_b", p.pdev);
        }
        if (!c->pdevs[p.pdev]->submit(p.p, xp, N, outp[i])) {
            // drain what was started: the staging buffers must not be released under a running device
            for (auto & q : c->pdevs) {
                q->wait();
            }
            return false;
        }
        t_submit[p.pdev] += ms_since(ts);
        if (N == 1) {
            SGLX_TRACE("sub_e", p.pdev);
        }
        if (c->serial || measuring) {
            c->pdevs[p.pdev]->wait();
            t_done[p.pdev] = ms_since(t0);
            t_alone[p.pdev] += ms_since(ts);
            n_rows[p.pdev] += (double) (p.r1 - p.r0);
        }
    }
    bool ok = true;
    for (size_t j = 0; j < c->pdevs.size(); ++j) {
        bool used = false;
        for (const auto & p : rec->parts) {
            used = used || p.pdev == (int) j;
        }
        if (used && !c->serial && !measuring) {
            ok = c->pdevs[j]->wait() && ok;
            t_done[j] = ms_since(t0);
            if (N == 1) {
                SGLX_TRACE("wait_e", (int64_t) j);
            }
        }
    }
    if (!ok) {
        return false;
    }

    if (measuring) {
        autotune_note(c, t_alone, n_rows);
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
    if (c->profile && !measuring && ++c->prof_calls > 3 + (uint64_t) (N == 1 ? 0 : 0)) { // the first calls compile shaders and pay the first use
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
    // inputs that the scheduler wants copied into this backend (cpy_tensor_async): the copy waits for the producing device, so it is done
    // only when a node of the next graph_compute reads the tensor. A ubatch without outputs gets an input for an output projection that is
    // never computed (llama-bench pp: every ubatch but the last), and copying it would stall the host until the previous device finished,
    // which is what stops the next ubatch from overlapping with it.
    struct pending_copy {
        ggml_backend_t       src_backend;
        const ggml_tensor *  src;
        ggml_tensor *        dst;
    };
    std::vector<pending_copy> pending;
};


const char * backend_get_name(ggml_backend_t) {
    return "SGLX0";
}

void backend_free(ggml_backend_t backend) {
    delete (backend_ctx *) backend->context;
    delete backend;
}

bool backend_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    if (backend_dst == nullptr || backend_dst->iface.free != backend_free || g_c == nullptr || !g_c->lazy_copy) {
        return false;
    }
    ((backend_ctx *) backend_dst->context)->pending.push_back({ backend_src, src, dst });
    return true;
}

void backend_synchronize(ggml_backend_t) {
    // every op is finished when graph_compute returns
}

enum ggml_status backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    composite * c = ((backend_ctx *) backend->context)->c;
    if (!ensure_init(c)) {
        return GGML_STATUS_FAILED;
    }
    {
        // the copies of inputs that a node of this graph reads (see backend_ctx::pending)
        const auto t_pc0 = clock_type::now();
        backend_ctx * bx = (backend_ctx *) backend->context;
        const bool had_pending = !bx->pending.empty();
        std::vector<ggml_backend_t> to_sync;
        for (const auto & pc : bx->pending) {
            bool used = false;
            for (int i = 0; i < cgraph->n_nodes && !used; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_is_empty(node) || ggml_op_is_empty(node->op)) {
                    continue;
                }
                for (int k = 0; k < GGML_MAX_SRC; ++k) {
                    used = used || (node->src[k] != nullptr && (node->src[k] == pc.dst || node->src[k]->view_src == pc.dst));
                }
            }
            if (used) {
                // a device with asynchronous reads and a destination in pinned memory: queue the read, wait once for all of them
                ggml_backend_buffer_t db = pc.dst->view_src != nullptr ? pc.dst->view_src->buffer : pc.dst->buffer;
                const ggml_tensor * sv = pc.src->view_src != nullptr ? pc.src->view_src : pc.src;
                if (moe::enabled() && pc.src_backend != nullptr && pc.src_backend->iface.get_tensor_async != nullptr && db != nullptr && buffer_owned(db) &&
                    bctx_of(db)->pinned != nullptr && sv->buffer != nullptr && !ggml_backend_buffer_is_host(sv->buffer) && ggml_are_same_layout(pc.src, pc.dst)) {
                    ggml_backend_tensor_get_async(pc.src_backend, pc.src, pc.dst->data, 0, ggml_nbytes(pc.src));
                    if (std::find(to_sync.begin(), to_sync.end(), pc.src_backend) == to_sync.end()) {
                        to_sync.push_back(pc.src_backend);
                    }
                } else {
                    // what the scheduler would have done in place of the lazy copy: wait for the device that produced the tensor, then copy
                    ggml_backend_synchronize(pc.src_backend);
                    ggml_backend_tensor_copy(const_cast<ggml_tensor *>(pc.src), pc.dst);
                }
            }
        }
        for (ggml_backend_t b : to_sync) {
            ggml_backend_synchronize(b);
        }
        bx->pending.clear();
        if (c->profile && moe::enabled() && had_pending) {
            moe::note_input_copy_ms(ms_since(t_pc0));
        }
    }
    const auto t_gc0 = clock_type::now();
    struct gc_timer {
        composite * c; clock_type::time_point t0;
        int n_nodes;
        ~gc_timer() {
            if (c->profile) {
                c->gc_big_ms.add(ms_since(t0));
                if (c->trace) {
                    fprintf(stderr, "SGLX trace: graph_compute %d nodes %.3f ms (t=%.3f)\n", n_nodes, ms_since(t0), std::chrono::duration<double, std::milli>(clock_type::now().time_since_epoch()).count());
                }
            }
        }
    } gc_timer_guard{ c, t_gc0, cgraph->n_nodes };
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_is_empty(node) || ggml_op_is_empty(node->op)) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_GET_ROWS: {
                const auto t0 = clock_type::now();
                if (c->profile && node->src[1]->ne[0] == 1 && c->t_mm_end.time_since_epoch().count() != 0) {
                    c->gap_host.add(std::chrono::duration<double, std::milli>(t0 - c->t_mm_end).count());
                }
                if (!do_get_rows(node)) {
                    return GGML_STATUS_FAILED;
                }
                c->t_gr_end = clock_type::now();
                if (c->profile && node->src[1]->ne[0] > 1) {
                    c->gr_big_ms.add(std::chrono::duration<double, std::milli>(c->t_gr_end - t0).count());
                }
                if (c->profile && node->src[1]->ne[0] == 1) {
                    c->gr_ms.add(std::chrono::duration<double, std::milli>(c->t_gr_end - t0).count());
                }
                break;
            }
            case GGML_OP_MUL_MAT:
                if (c->profile && node->src[1]->ne[1] == 1 && c->t_gr_end.time_since_epoch().count() != 0) {
                    {
                        const double g = std::chrono::duration<double, std::milli>(clock_type::now() - c->t_gr_end).count();
                        if (g < 100.0) { c->gap_layers.add(g); } // not the shader compilation of the first tokens
                    }
                }
                if (!do_mul_mat(c, node)) {
                    return GGML_STATUS_FAILED;
                }
                c->t_mm_end = clock_type::now();
                break;
            default:
                if (moe::handles(node)) {
                    if (!moe::exec(cgraph, i)) {
                        SGLX_LOG("MoE op %s (%s) failed\n", ggml_op_desc(node), node->name);
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                SGLX_LOG("op %s is not supported\n", ggml_op_desc(node));
                return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

// Events: every op of this backend is finished when graph_compute returns, so an event is already signalled when it is recorded.
// They exist so that ggml's scheduler may use pipeline parallelism with the other devices (llama.cpp turns it on only when every device has events).
void backend_event_record(ggml_backend_t, ggml_backend_event_t) {}
void backend_event_wait(ggml_backend_t, ggml_backend_event_t) {}

const ggml_backend_i backend_i = {
    /* .get_name                = */ backend_get_name,
    /* .free                    = */ backend_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ backend_cpy_tensor_async,
    /* .synchronize             = */ backend_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ backend_graph_compute,
    /* .event_record            = */ backend_event_record,
    /* .event_wait              = */ backend_event_wait,
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
        /* .async                = */ true,  // graph_compute is synchronous, so "async" is trivially true (needed for pipeline parallelism)
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ moe::enabled(),   // MoE mode: the experts stay in the mmap of the model file
        /* .events               = */ true,
        /* .mmap_support         = */ moe::enabled(),
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

ggml_backend_buffer_t dev_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t) {
    if (!moe::enabled() || ptr == nullptr) {
        return nullptr;
    }
#if defined(__linux__)
    // map the pages of the experts now (they are in the page cache or are read from disk once): the first access of a page costs a minor fault, which is as much
    // as the mat-vec on the page (0.3 ms per expert) for the first token that selects the expert. GGML_SGLX_MOE_PREFAULT=0 leaves it to the first use.
    if (getenv("GGML_SGLX_MOE_PREFAULT") == nullptr || atoi(getenv("GGML_SGLX_MOE_PREFAULT")) != 0) {
        const auto t0 = clock_type::now();
        const uintptr_t a0 = ((uintptr_t) ptr + 4095) & ~(uintptr_t) 4095;
        const uintptr_t a1 = ((uintptr_t) ptr + size) & ~(uintptr_t) 4095;
        if (a1 > a0 && madvise((void *) a0, a1 - a0, 22 /* MADV_POPULATE_READ */) == 0) {
            SGLX_LOG("mapped %.1f GiB of experts in %.1f s\n", (double) (a1 - a0) / 1073741824.0, ms_since(t0) / 1000.0);
        }
    }
#endif
    auto * bc = new buffer_ctx();
    bc->base = (char *) ptr;
    bc->size = size;
    bc->external = true;
    return ggml_backend_buffer_init(&dev_c(dev)->buft, buffer_i, bc, size);
}

ggml_backend_event_t dev_event_new(ggml_backend_dev_t dev) {
    return new ggml_backend_event{ /* .device = */ dev, /* .context = */ nullptr };
}

void dev_event_free(ggml_backend_dev_t, ggml_backend_event_t event) {
    delete event;
}

void dev_event_synchronize(ggml_backend_dev_t, ggml_backend_event_t) {}

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
            return !c->hostbuf && get_rows_shape_ok(c, op);
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_GLU:
        case GGML_OP_MUL:
        case GGML_OP_ADD:
            return moe::enabled() && moe::supports_op(op);
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
    /* .buffer_from_host_ptr = */ dev_buffer_from_host_ptr,
    /* .supports_op          = */ dev_supports_op,
    /* .supports_buft        = */ dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ dev_event_new,
    /* .event_free           = */ dev_event_free,
    /* .event_synchronize    = */ dev_event_synchronize,
};

} // namespace

ggml_backend_dev_t find_ggml_device(const char * name) {
    return find_ggml_device_impl(name);
}

// ---- registration ----

void attach(ggml_backend_reg_t reg, const std::vector<std::pair<std::string, cl11_device *>> & cl11) {
    const char * env = getenv("GGML_SGLX_SPLIT");
    const bool moe_only = (env == nullptr || *env == '\0') && moe::enabled();
    if (((env == nullptr || *env == '\0') && !moe_only) || g_c != nullptr) {
        return;
    }
    if (moe_only) {
        env = "";
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
    if (c->spec.empty() && !moe_only) {
        delete c;
        return;
    }
    c->threaded_last = getenv("GGML_SGLX_THREADED_LAST") != nullptr && atoi(getenv("GGML_SGLX_THREADED_LAST")) != 0;
    c->profile = getenv("GGML_SGLX_PROFILE") != nullptr && atoi(getenv("GGML_SGLX_PROFILE")) != 0;
    c->lazy_copy = getenv("GGML_SGLX_LAZY_COPY") == nullptr || atoi(getenv("GGML_SGLX_LAZY_COPY")) != 0;
    c->hostbuf = getenv("GGML_SGLX_HOSTBUF") != nullptr && atoi(getenv("GGML_SGLX_HOSTBUF")) != 0;
    c->trace = getenv("GGML_SGLX_TRACE") != nullptr && atoi(getenv("GGML_SGLX_TRACE")) != 0;
    c->serial = getenv("GGML_SGLX_SERIAL") != nullptr && atoi(getenv("GGML_SGLX_SERIAL")) != 0;
    if (getenv("GGML_SGLX_AUTOTUNE") != nullptr) {
        const char * at = getenv("GGML_SGLX_AUTOTUNE");
        c->at_rounds = atoi(at);
        const char * comma = strchr(at, ',');
        if (comma != nullptr && atoi(comma + 1) > 0) {
            c->at_calls = atoi(comma + 1);
        }
    }
    if (getenv("GGML_SGLX_PROFILE_EVERY") != nullptr) {
        c->profile_every = atoi(getenv("GGML_SGLX_PROFILE_EVERY"));
    }
    c->desc = "SGLX composite (" + (s.empty() ? std::string("MoE") : s) + ")";
    c->buft = ggml_backend_buffer_type{ /* .iface = */ buft_i, /* .device = */ &c->dev, /* .context = */ c };
    c->dev = ggml_backend_device{ /* .iface = */ dev_i, /* .reg = */ reg, /* .context = */ c };
    g_c = c;
    if (moe::enabled()) {
        moe::init(&c->buft, moe::hooks{ buffer_owned });
    }
    atexit(print_profile_atexit);
    SGLX_TRACE_DUMP_ATEXIT();
}

size_t device_count() {
    return g_c != nullptr ? 1 : 0;
}

ggml_backend_dev_t device_get(size_t index) {
    GGML_ASSERT(g_c != nullptr && index == 0);
    return &g_c->dev;
}

} // namespace sglx
