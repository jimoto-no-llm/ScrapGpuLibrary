// pdev over a device of the ScrapGpu C API that lives in another process: the "remote" backend (src/backends/remote) and a scrapgpu-device-server.
// Named "REMOTE<i>" in GGML_SGLX_SPLIT, i = the index in the device list of the servers that this process started or connected to.
//
// Why: the Fermi GT 430 cannot share a process with the Kepler cards under the NVIDIA 390 driver (OpenCL and CUDA alike). The server process opens
// it alone, with the CL11 backend (--backend opencl11) or the CUDA backend (--backend cuda, sm_20 PTX); this pdev drives it through the C API:
// tensor_upload of the rows (the server repacks them to R32), buffer_write of the input columns, op_execute(MUL_MAT), buffer_read of the result.
//
// The servers (one per process, created on the first REMOTE<i> of the split):
//
//   GGML_SGLX_REMOTE_SOCKET=/path/to.sock   connect to a scrapgpu-device-server that is already running (--socket PATH --backend ...)
//   otherwise a server is started (fork + exec, scrapgpu_remote_spawn; it stops when this process disconnects or dies):
//     GGML_SGLX_REMOTE_BACKEND=opencl11|cuda     the --backend of the server (default opencl11)
//     GGML_SGLX_REMOTE_ENV="K=V;K=V"             environment overrides of the server, e.g.
//                                                "GGML_CL11_PLATFORMS=NVIDIA;GGML_CL11_DEVICES=GT 430"     (CL11)
//                                                "SCRAPGPU_CUDA_DEVICES=GT 430"                           (CUDA)
//     SCRAPGPU_DEVICE_SERVER=/path/scrapgpu-device-server  the binary (default: next to the executable, then PATH)
//   GGML_SGLX_REMOTE_BATCH=0                   one round trip per call (default: the write, the op and the read of a job go in one round trip)
//
// A remote call is a round trip on a socket and returns when the server is done, so submit() hands the job to a worker thread (the host thread goes
// on to submit the other devices) and wait() waits for it. One job is: buffer_write(x), op_execute, buffer_read(result) = 3 calls, one round trip with
// scrapgpu_remote_set_batching (3 round trips without it); the
// weights stay in the server's memory in the R32 layout. The worker is the only thread that calls into the connection after the upload.
//
// Built only with SGLX_WITH_REMOTE (adapters/ggml/CMakeLists.txt option CL11_WITH_REMOTE, POSIX only); otherwise make_pdev_scrapgpu() returns nullptr.

#include "composite.h"
#include "pdev.h"

#ifdef SGLX_WITH_REMOTE

#include "scrapgpu/api.h"
#include "scrapgpu/backend.h"
#include "scrapgpu/buffer.h"
#include "scrapgpu/device.h"
#include "scrapgpu/op.h"
#include "scrapgpu/remote.h"
#include "scrapgpu/stream.h"
#include "sgl_trace.h"
#include "scrapgpu/tensor.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace sglx {
namespace {

using clock_type = std::chrono::steady_clock;

double ms_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

scrapgpu_dtype dtype_of(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return SCRAPGPU_DTYPE_F32;
        case GGML_TYPE_F16:  return SCRAPGPU_DTYPE_F16;
        case GGML_TYPE_Q4_0: return SCRAPGPU_DTYPE_Q4_0;
        case GGML_TYPE_Q8_0: return SCRAPGPU_DTYPE_Q8_0;
        case GGML_TYPE_Q4_K: return SCRAPGPU_DTYPE_Q4_K;
        case GGML_TYPE_Q6_K: return SCRAPGPU_DTYPE_Q6_K;
        default:             return SCRAPGPU_DTYPE_UNKNOWN;
    }
}

std::string last_error() {
    const char * e = scrapgpu_last_error();
    return e != nullptr ? e : "";
}

// a contiguous tensor of `dtype` and shape ne0 x ne1 in `buffer`
scrapgpu_tensor_desc make_desc(scrapgpu_dtype dtype, int64_t ne0, int64_t ne1, scrapgpu_buffer_t * buffer, scrapgpu_layout layout) {
    scrapgpu_tensor_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.dtype = dtype;
    d.ne[0] = ne0;
    d.ne[1] = ne1;
    d.ne[2] = 1;
    d.ne[3] = 1;
    d.nb[0] = scrapgpu_dtype_type_size(dtype);
    d.nb[1] = scrapgpu_tensor_row_size(dtype, ne0);
    d.nb[2] = d.nb[1] * (uint64_t) ne1;
    d.nb[3] = d.nb[2];
    d.buffer = buffer;
    d.offset = 0;
    d.layout = layout;
    return d;
}

// ---- the servers of this process ----

struct remote_set {
    std::mutex                    mtx;
    bool                          tried = false;
    scrapgpu_remote_t *           remote = nullptr;
    std::vector<scrapgpu_device_t *> devices;
};

remote_set & servers() {
    static remote_set * s = new remote_set(); // never destroyed: the pdevs live until exit
    return *s;
}

void close_servers_atexit() {
    remote_set & s = servers();
    sgr::trace_dump(".cli");
    if (s.remote != nullptr) {
        scrapgpu_remote_close(s.remote);
        s.remote = nullptr;
    }
}

std::vector<std::string> split_env(const char * spec) {
    std::vector<std::string> out;
    std::string s(spec != nullptr ? spec : "");
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t end = s.find(';', pos);
        if (end == std::string::npos) {
            end = s.size();
        }
        if (end > pos) {
            out.push_back(s.substr(pos, end - pos));
        }
        pos = end + 1;
    }
    return out;
}

// start (or connect to) the server once and list its devices
bool ensure_servers() {
    remote_set & s = servers();
    std::lock_guard<std::mutex> lock(s.mtx);
    if (s.tried) {
        return s.remote != nullptr;
    }
    s.tried = true;
    scrapgpu_status st;
    const char * sock = getenv("GGML_SGLX_REMOTE_SOCKET");
    if (sock != nullptr && *sock != '\0') {
        st = scrapgpu_remote_connect(sock, &s.remote);
    } else {
        std::vector<std::string> env = split_env(getenv("GGML_SGLX_REMOTE_ENV"));
        const char * be = getenv("GGML_SGLX_REMOTE_BACKEND");
        env.push_back(std::string("SCRAPGPU_REMOTE_BACKEND=") + (be != nullptr && *be != '\0' ? be : "opencl11"));
        std::vector<const char *> ptrs;
        for (const auto & e : env) {
            ptrs.push_back(e.c_str());
        }
        ptrs.push_back(nullptr);
        st = scrapgpu_remote_spawn(ptrs.data(), &s.remote);
    }
    if (st != SCRAPGPU_STATUS_OK || s.remote == nullptr) {
        fprintf(stderr, "SGLX: cannot start or reach a scrapgpu-device-server: %s\n", last_error().c_str());
        s.remote = nullptr;
        return false;
    }
    // write x + op_execute + read the result: one round trip instead of three (the host is busy with the other devices, every round trip is two
    // process wake-ups); GGML_SGLX_REMOTE_BATCH=0 turns it off
    const char * bt = getenv("GGML_SGLX_REMOTE_BATCH");
    if (bt == nullptr || atoi(bt) != 0) {
        scrapgpu_remote_set_batching(s.remote, 1);
    }
    atexit(close_servers_atexit);
    const size_t n = scrapgpu_get_device_count();
    for (size_t i = 0; i < n; ++i) {
        scrapgpu_device_t * d = nullptr;
        if (scrapgpu_get_device(i, &d) == SCRAPGPU_STATUS_OK && d != nullptr) {
            s.devices.push_back(d);
        }
    }
    return true;
}

// ---- the pdev ----

struct remote_piece : piece {
    scrapgpu_dtype dt = SCRAPGPU_DTYPE_UNKNOWN;
    int64_t        K = 0;
    int64_t        rows = 0;
    scrapgpu_layout layout = SCRAPGPU_LAYOUT_NATURAL;
    scrapgpu_buffer_t * w = nullptr;
};

struct job {
    remote_piece * p;
    const float *  x;
    int64_t        N;
    float *        out;
};

class pdev_scrapgpu final : public pdev {
public:
    pdev_scrapgpu(scrapgpu_device_t * dev, int index) : dev_(dev) {
        name_ = "REMOTE" + std::to_string(index);
        scrapgpu_device_info info;
        scrapgpu_device_info_init(&info, sizeof(info));
        if (scrapgpu_device_get_info(dev, &info) == SCRAPGPU_STATUS_OK && info.name != nullptr) {
            desc_ = info.name;
            if (desc_.rfind("remote:", 0) == 0) {
                desc_ = desc_.substr(7);
            }
            backend_ = info.backend_name != nullptr ? info.backend_name : "";
        }
        weight_ = default_weight_for(desc_, false);
        if (scrapgpu_stream_create(dev_, &stream_) != SCRAPGPU_STATUS_OK) {
            stream_ = nullptr;
            return;
        }
        worker_ = std::thread([this] { run(); });
    }

    ~pdev_scrapgpu() override {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_job_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
        free_work_buffers();
        if (stream_ != nullptr) {
            scrapgpu_stream_free(stream_);
        }
    }

    bool ok() const { return stream_ != nullptr; }
    bool threaded() const override { return true; }

    const std::string & name() const override { return name_; }
    const std::string & description() const override { return desc_; }
    double default_weight() const override { return weight_; }

    bool supports_mul_mat(ggml_type type) override {
        auto it = sup_.find(type);
        if (it != sup_.end()) {
            return it->second;
        }
        bool ok = false;
        const scrapgpu_dtype dt = dtype_of(type);
        if (dt != SCRAPGPU_DTYPE_UNKNOWN && stream_ != nullptr) {
            const int64_t K = 512, rows = 64;
            drain();
            std::lock_guard<std::mutex> lock(conn_);
            scrapgpu_tensor_desc probe = make_desc(dt, K, rows, nullptr, SCRAPGPU_LAYOUT_NATURAL);
            const scrapgpu_layout layout = scrapgpu_tensor_r32_eligible(&probe) ? SCRAPGPU_LAYOUT_R32 : SCRAPGPU_LAYOUT_NATURAL;
            scrapgpu_buffer_t *wb = nullptr, *xb = nullptr, *ob = nullptr;
            if (scrapgpu_buffer_alloc(dev_, (size_t) probe.nb[2], &wb) == SCRAPGPU_STATUS_OK &&
                scrapgpu_buffer_alloc(dev_, (size_t) K * 4, &xb) == SCRAPGPU_STATUS_OK &&
                scrapgpu_buffer_alloc(dev_, (size_t) rows * 4, &ob) == SCRAPGPU_STATUS_OK) {
                scrapgpu_tensor_desc w = make_desc(dt, K, rows, wb, layout);
                scrapgpu_tensor_desc x = make_desc(SCRAPGPU_DTYPE_F32, K, 1, xb, SCRAPGPU_LAYOUT_NATURAL);
                scrapgpu_tensor_desc o = make_desc(SCRAPGPU_DTYPE_F32, rows, 1, ob, SCRAPGPU_LAYOUT_NATURAL);
                scrapgpu_tensor_desc ins[2] = { w, x };
                scrapgpu_mul_mat_params prm = { sizeof(prm), 0 };
                scrapgpu_op_desc od;
                memset(&od, 0, sizeof(od));
                od.struct_size = sizeof(od);
                od.op = SCRAPGPU_OP_MUL_MAT;
                od.inputs = ins;
                od.input_count = 2;
                od.outputs = &o;
                od.output_count = 1;
                od.params = &prm;
                od.params_size = sizeof(prm);
                const char * reason = nullptr;
                ok = scrapgpu_op_supported(dev_, &od, &reason) == SCRAPGPU_STATUS_OK;
            }
            if (wb != nullptr) scrapgpu_buffer_free(wb);
            if (xb != nullptr) scrapgpu_buffer_free(xb);
            if (ob != nullptr) scrapgpu_buffer_free(ob);
        }
        sup_[type] = ok;
        return ok;
    }

    piece * piece_create(ggml_type type, int64_t K, int64_t rows) override {
        const scrapgpu_dtype dt = dtype_of(type);
        if (dt == SCRAPGPU_DTYPE_UNKNOWN || stream_ == nullptr || K <= 0 || rows <= 0) {
            return nullptr;
        }
        scrapgpu_tensor_desc probe = make_desc(dt, K, rows, nullptr, SCRAPGPU_LAYOUT_NATURAL);
        auto * p = new remote_piece();
        p->dt = dt;
        p->K = K;
        p->rows = rows;
        p->layout = scrapgpu_tensor_r32_eligible(&probe) ? SCRAPGPU_LAYOUT_R32 : SCRAPGPU_LAYOUT_NATURAL;
        std::lock_guard<std::mutex> lock(conn_);
        if (scrapgpu_buffer_alloc(dev_, (size_t) probe.nb[2], &p->w) != SCRAPGPU_STATUS_OK) {
            fprintf(stderr, "SGLX: %s: cannot allocate %.1f MiB: %s\n", name_.c_str(), (double) probe.nb[2] / 1048576.0, last_error().c_str());
            delete p;
            return nullptr;
        }
        return p;
    }

    bool piece_upload(piece * pp, const void * data, size_t bytes) override {
        auto * p = static_cast<remote_piece *>(pp);
        const scrapgpu_tensor_desc d = make_desc(p->dt, p->K, p->rows, p->w, p->layout);
        if (bytes != (size_t) d.nb[2]) {
            return false;
        }
        drain();
        std::lock_guard<std::mutex> lock(conn_);
        if (scrapgpu_tensor_upload(stream_, &d, data, bytes) != SCRAPGPU_STATUS_OK ||
            scrapgpu_stream_sync(stream_) != SCRAPGPU_STATUS_OK) {
            fprintf(stderr, "SGLX: %s: upload failed: %s\n", name_.c_str(), last_error().c_str());
            return false;
        }
        return true;
    }

    void piece_free(piece * pp) override {
        auto * p = static_cast<remote_piece *>(pp);
        drain();
        {
            std::lock_guard<std::mutex> lock(conn_);
            if (p->w != nullptr) {
                scrapgpu_buffer_free(p->w);
            }
        }
        delete p;
    }

    bool submit(piece * pp, const float * x, int64_t N, float * out) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (failed_) {
            return false;
        }
        q_.push_back(job{ static_cast<remote_piece *>(pp), x, N, out });
        ++outstanding_;
        sgr::trace("p_submit");
        cv_job_.notify_one();
        return true;
    }

    bool wait() override {
        std::unique_lock<std::mutex> lock(mu_);
        sgr::trace("p_wait_b");
        cv_done_.wait(lock, [this] { return outstanding_ == 0; });
        sgr::trace("p_wait_e");
        const bool ok = !failed_;
        failed_ = false;
        return ok;
    }

    std::string stats() override {
        std::lock_guard<std::mutex> lock(mu_);
        if (st_.jobs == 0) {
            return std::string();
        }
        char buf[260];
        snprintf(buf, sizeof(buf), "; %s worker per job (%llu jobs): write x %.3f ms, op_execute %.3f ms, read result (waits for the GPU) %.3f ms, total %.3f ms",
                 backend_.c_str(), (unsigned long long) st_.jobs, st_.write / st_.jobs, st_.exec / st_.jobs, st_.read / st_.jobs, st_.total / st_.jobs);
        return buf;
    }

private:
    // wait for the worker to be idle (the jobs of a missing wait() included)
    void drain() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_done_.wait(lock, [this] { return outstanding_ == 0; });
    }

    // GGML_SGLX_REMOTE_CPU=n pins the worker to cpu n, GGML_SGLX_REMOTE_SCHED=batch|idle gives it that scheduling policy (Linux, no privilege): a
    // woken SCHED_NORMAL worker preempts the host thread that submitted the job, which then submits the other devices late
    static void apply_worker_sched() {
#ifdef __linux__
        const char * cpu = getenv("GGML_SGLX_REMOTE_CPU");
        if (cpu != nullptr && *cpu != '\0') {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(atoi(cpu), &set);
            if (sched_setaffinity(0, sizeof(set), &set) != 0) {
                fprintf(stderr, "SGLX: GGML_SGLX_REMOTE_CPU failed\n");
            }
        }
        const char * pol = getenv("GGML_SGLX_REMOTE_SCHED");
        if (pol != nullptr && *pol != '\0') {
            sched_param sp = {};
            const int policy = std::string(pol) == "batch" ? SCHED_BATCH : std::string(pol) == "idle" ? SCHED_IDLE : SCHED_OTHER;
            if (sched_setscheduler(0, policy, &sp) != 0) {
                fprintf(stderr, "SGLX: GGML_SGLX_REMOTE_SCHED failed\n");
            }
        }
#endif
    }

    void run() {
        apply_worker_sched();
        std::unique_lock<std::mutex> lock(mu_);
        for (;;) {
            cv_job_.wait(lock, [this] { return stop_ || !q_.empty(); });
            if (stop_ && q_.empty()) {
                return;
            }
            job j = q_.front();
            q_.pop_front();
            lock.unlock();
            sgr::trace("p_wake");
            const bool ok = execute(j);
            sgr::trace("p_exec_e");
            lock.lock();
            if (!ok) {
                failed_ = true;
            }
            --outstanding_;
            if (outstanding_ == 0) {
                cv_done_.notify_all();
            }
        }
    }

    // device buffers for the input columns and the result of at least xbytes / obytes (grown with headroom); worker thread only
    bool grow(size_t xbytes, size_t obytes) {
        if (xbytes > xcap_) {
            const size_t cap = std::max<size_t>(xbytes, xcap_ * 2);
            if (xb_ != nullptr) {
                scrapgpu_buffer_free(xb_);
                xb_ = nullptr;
                xcap_ = 0;
            }
            if (scrapgpu_buffer_alloc(dev_, cap, &xb_) != SCRAPGPU_STATUS_OK) {
                xb_ = nullptr;
                return false;
            }
            xcap_ = cap;
        }
        if (obytes > ocap_) {
            const size_t cap = std::max<size_t>(obytes, ocap_ * 2);
            if (ob_ != nullptr) {
                scrapgpu_buffer_free(ob_);
                ob_ = nullptr;
                ocap_ = 0;
            }
            if (scrapgpu_buffer_alloc(dev_, cap, &ob_) != SCRAPGPU_STATUS_OK) {
                ob_ = nullptr;
                return false;
            }
            ocap_ = cap;
        }
        return true;
    }

    void free_work_buffers() {
        if (xb_ != nullptr) {
            scrapgpu_buffer_free(xb_);
            xb_ = nullptr;
        }
        if (ob_ != nullptr) {
            scrapgpu_buffer_free(ob_);
            ob_ = nullptr;
        }
        xcap_ = ocap_ = 0;
    }

    bool execute(const job & j) {
        remote_piece * p = j.p;
        const size_t xbytes = (size_t) p->K * (size_t) j.N * sizeof(float);
        const size_t obytes = (size_t) p->rows * (size_t) j.N * sizeof(float);
        const auto t0 = clock_type::now();
        std::lock_guard<std::mutex> lock(conn_);
        if (!grow(xbytes, obytes)) {
            fprintf(stderr, "SGLX: %s: cannot allocate the work buffers: %s\n", name_.c_str(), last_error().c_str());
            return false;
        }
        if (scrapgpu_buffer_write(stream_, xb_, 0, j.x, xbytes, 0) != SCRAPGPU_STATUS_OK) {
            fprintf(stderr, "SGLX: %s: buffer_write failed: %s\n", name_.c_str(), last_error().c_str());
            return false;
        }
        const double t_write = ms_since(t0);
        scrapgpu_tensor_desc ins[2] = { make_desc(p->dt, p->K, p->rows, p->w, p->layout), make_desc(SCRAPGPU_DTYPE_F32, p->K, j.N, xb_, SCRAPGPU_LAYOUT_NATURAL) };
        scrapgpu_tensor_desc o = make_desc(SCRAPGPU_DTYPE_F32, p->rows, j.N, ob_, SCRAPGPU_LAYOUT_NATURAL);
        scrapgpu_mul_mat_params prm = { sizeof(prm), 0 };
        scrapgpu_op_desc od;
        memset(&od, 0, sizeof(od));
        od.struct_size = sizeof(od);
        od.op = SCRAPGPU_OP_MUL_MAT;
        od.inputs = ins;
        od.input_count = 2;
        od.outputs = &o;
        od.output_count = 1;
        od.params = &prm;
        od.params_size = sizeof(prm);
        if (scrapgpu_op_execute(stream_, &od) != SCRAPGPU_STATUS_OK) {
            fprintf(stderr, "SGLX: %s: op_execute(MUL_MAT) failed: %s\n", name_.c_str(), last_error().c_str());
            return false;
        }
        sgr::trace("p_op_e");
        const double t_exec = ms_since(t0) - t_write;
        // blocking: the stream is in order, so this returns when the kernel is done and the result is in `out`
        if (scrapgpu_buffer_read(stream_, ob_, 0, j.out, obytes, 1) != SCRAPGPU_STATUS_OK) {
            fprintf(stderr, "SGLX: %s: buffer_read failed: %s\n", name_.c_str(), last_error().c_str());
            return false;
        }
        const double t_all = ms_since(t0);
        std::lock_guard<std::mutex> sl(mu_);
        st_.jobs++;
        st_.write += t_write;
        st_.exec += t_exec;
        st_.read += t_all - t_write - t_exec;
        st_.total += t_all;
        return true;
    }

    struct stats_t {
        uint64_t jobs = 0;
        double   write = 0, exec = 0, read = 0, total = 0;
    };

    scrapgpu_device_t *    dev_ = nullptr;
    scrapgpu_stream_t *    stream_ = nullptr;
    std::string            name_;
    std::string            desc_;
    std::string            backend_;
    double                 weight_ = 1.0;
    std::map<ggml_type, bool> sup_;

    std::mutex             conn_; // the connection: uploads from the host thread, jobs from the worker
    std::mutex             mu_;   // the queue
    std::condition_variable cv_job_, cv_done_;
    std::deque<job>        q_;
    size_t                 outstanding_ = 0;
    bool                   stop_ = false;
    bool                   failed_ = false;
    std::thread            worker_;
    stats_t                st_;

    scrapgpu_buffer_t *    xb_ = nullptr;
    scrapgpu_buffer_t *    ob_ = nullptr;
    size_t                 xcap_ = 0;
    size_t                 ocap_ = 0;
};

} // namespace

std::unique_ptr<pdev> make_pdev_scrapgpu(int index) {
    if (index < 0 || !ensure_servers()) {
        return nullptr;
    }
    remote_set & s = servers();
    if ((size_t) index >= s.devices.size()) {
        fprintf(stderr, "SGLX: the device server has %zu device(s), no REMOTE%d\n", s.devices.size(), index);
        return nullptr;
    }
    auto p = std::make_unique<pdev_scrapgpu>(s.devices[(size_t) index], index);
    if (!p->ok()) {
        return nullptr;
    }
    return p;
}

} // namespace sglx

#else // !SGLX_WITH_REMOTE

namespace sglx {

std::unique_ptr<pdev> make_pdev_scrapgpu(int) {
    return nullptr;
}

} // namespace sglx

#endif
