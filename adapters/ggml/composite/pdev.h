// SGLX composite device: the physical-device interface (pdev).
//
// A pdev is one physical GPU (or the CPU) that holds a slice of the rows of a weight matrix and computes
// MUL_MAT for that slice. The composite device (composite_device.cpp) splits a weight along its rows
// (the vocabulary of the output projection), hands every pdev its rows once (piece_upload) and then, for
// every MUL_MAT, submits the same input columns to all pdevs before it waits for any of them, so the
// devices run at the same time although the host is a single thread.
//
// Implementations: pdev_ggml.cpp (any ggml backend device through the public ggml-backend.h API: the
// official Vulkan backend, ggml-cpu, Metal) and pdev_cl11.cpp (a CL11 OpenCL 1.1 device of this plugin).
// docs/plans/MOE_COMPOSITE_DEVICE.md chapter 6.2 describes the full design; this is the part that the
// first version (the output projection) needs.

#ifndef SGLX_PDEV_H
#define SGLX_PDEV_H

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace sglx {

// the rows of one weight that live on one pdev (plus the activation scratch it needs), opaque for the composite
struct piece {
    virtual ~piece() = default;
};

class pdev {
public:
    virtual ~pdev() = default;

    // the ggml device name ("Vulkan0", "CL11_1", "CPU") and a description
    virtual const std::string & name() const = 0;
    virtual const std::string & description() const = 0;

    // relative speed of a quantized mat-vec on this device (about the effective GB/s); the default split ratio
    virtual double default_weight() const = 0;

    // can the pdev compute MUL_MAT(weight of `type`, F32 columns) (any number of columns)? Cached.
    virtual bool supports_mul_mat(ggml_type type) = 0;

    // room for `rows` rows of K columns of `type` on the device; nullptr when the allocation fails
    virtual piece * piece_create(ggml_type type, int64_t K, int64_t rows) = 0;
    // `bytes` = rows * row size: the rows in the natural ggml layout, contiguous. Blocking.
    virtual bool piece_upload(piece * p, const void * data, size_t bytes) = 0;
    virtual void piece_free(piece * p) = 0;

    // out[c * rows + r] = sum_k W[r][k] * x[c * K + k] for c < N, r < rows. Starts the work and returns (the host
    // buffers `x` and `out` must stay valid until wait() returned).
    virtual bool submit(piece * p, const float * x, int64_t N, float * out) = 0;
    // blocks until everything submitted since the last wait() is finished and written to its `out`
    virtual bool wait() = 0;
    // GGML_SGLX_PROFILE: where submit() spent its time (empty when the pdev keeps no timers)
    virtual std::string stats() { return std::string(); }
};

// the effective mat-vec GB/s used for the default split, from a device description (substring table, see composite_device.cpp)
double default_weight_for(const std::string & description, bool is_cpu);

std::unique_ptr<pdev> make_pdev_ggml(ggml_backend_dev_t dev);

} // namespace sglx

#endif
