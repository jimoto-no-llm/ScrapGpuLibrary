#ifndef SCRAPGPU_REMOTE_TRACE_H
#define SCRAPGPU_REMOTE_TRACE_H

// Timestamp trace for latency work on the remote path (docs/benchmarks/moe-ep-20261023/gt430-latency). Off unless SGL_TRACE=<prefix> is set; then
// every sgr::trace(tag, a) stores {CLOCK_MONOTONIC ns, thread id, tag, a} in memory (no I/O on the hot path), and sgr::trace_dump(suffix) writes
// <prefix><suffix> as text lines "ns tid tag a". CLOCK_MONOTONIC is shared by the processes of a machine, so the client and the server files line up.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include <sys/syscall.h>
#include <unistd.h>

namespace sgr {

struct trace_ev {
    uint64_t    t;
    uint32_t    tid;
    uint32_t    pad;
    const char * tag;
    int64_t     a;
};

struct trace_state {
    bool                  on = false;
    trace_ev *            ev = nullptr;
    uint32_t              cap = 0;
    std::atomic<uint32_t> n{0};
    std::string           prefix;
};

inline trace_state & trace_st() {
    static trace_state * s = [] {
        auto * st = new trace_state();
        const char * p = std::getenv("SGL_TRACE");
        if (p != nullptr && *p != '\0') {
            st->prefix = p;
            st->cap    = 1u << 21;
            st->ev     = new trace_ev[st->cap];
            st->on     = true;
        }
        return st;
    }();
    return *s;
}

inline uint64_t trace_now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

inline void trace(const char * tag, int64_t a = 0) {
    trace_state & s = trace_st();
    if (!s.on) {
        return;
    }
    const uint32_t i = s.n.fetch_add(1, std::memory_order_relaxed);
    if (i >= s.cap) {
        return;
    }
    s.ev[i] = trace_ev{ trace_now_ns(), (uint32_t) syscall(SYS_gettid), 0, tag, a };
}

inline void trace_dump(const char * suffix) {
    trace_state & s = trace_st();
    if (!s.on) {
        return;
    }
    s.on = false;
    FILE * f = std::fopen((s.prefix + suffix).c_str(), "w");
    if (f == nullptr) {
        return;
    }
    const uint32_t n = s.n.load() < s.cap ? s.n.load() : s.cap;
    for (uint32_t i = 0; i < n; ++i) {
        std::fprintf(f, "%llu %u %s %lld\n", (unsigned long long) s.ev[i].t, s.ev[i].tid, s.ev[i].tag, (long long) s.ev[i].a);
    }
    std::fclose(f);
}

} // namespace sgr

#endif
