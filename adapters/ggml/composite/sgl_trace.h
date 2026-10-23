// SGL_TRACE=<prefix>: timestamp trace of the composite device (docs/benchmarks/moe-ep-20261023/gt430-latency); only in builds with SGLX_WITH_REMOTE
// (it shares the recorder of src/backends/remote/trace.h with the remote client). SGLX_TRACE(tag, arg) is a no-op otherwise.
#ifndef SGLX_SGL_TRACE_H
#define SGLX_SGL_TRACE_H

#ifdef SGLX_WITH_REMOTE
#include "remote/trace.h"
#define SGLX_TRACE(...) ::sgr::trace(__VA_ARGS__)
#define SGLX_TRACE_ON() (::sgr::trace_st().on)
#define SGLX_TRACE_DUMP_ATEXIT() do { if (::sgr::trace_st().on) { atexit([] { ::sgr::trace_dump(".cli"); }); } } while (0)
#else
#define SGLX_TRACE(...) ((void) 0)
#define SGLX_TRACE_ON() false
#define SGLX_TRACE_DUMP_ATEXIT() ((void) 0)
#endif

#endif
