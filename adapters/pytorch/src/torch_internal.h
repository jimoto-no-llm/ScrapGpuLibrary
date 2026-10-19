// Shared state of the PyTorch adapter translation units (torch_api.cpp, device_tensor.cpp).
#pragma once

#include "scrapgpu/buffer.h"
#include "scrapgpu/compat.h"
#include "scrapgpu/device.h"
#include "scrapgpu/op.h"
#include "scrapgpu/status.h"
#include "scrapgpu/stream.h"
#include "scrapgpu/tensor.h"

#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <mutex>
#include <string>

namespace sgtorch {

// All adapter entry points serialize on g_mu (caller holds it for the members below).
extern std::mutex g_mu;
extern std::string g_backend;
extern scrapgpu_compat_options g_compat;
extern scrapgpu_device_t * g_dev;
extern scrapgpu_stream_t * g_stream;

void check_status(scrapgpu_status st, const char * what);
// Caller holds g_mu: registers the backend and creates the device stream on first use.
void ensure_backend();
scrapgpu_tensor_desc make_desc(scrapgpu_dtype dt, scrapgpu_layout layout, int64_t n0, int64_t n1, int64_t n2, int64_t n3,
                               scrapgpu_buffer_t * buf);
at::Tensor require_cpu_f32(const at::Tensor & t, const char * name);

// A quantized / float weight [out_features, in_features] living in a device buffer.
class WeightImpl : public torch::CustomClassHolder {
public:
    WeightImpl(at::Tensor weight_f32, std::string quant_type);
    ~WeightImpl() override;
    const scrapgpu_tensor_desc & desc() const { return desc_; }
    int64_t in_features() const { return in_features_; }
    int64_t out_features() const { return out_features_; }
    const std::string & quant_type() const { return quant_type_; }

private:
    scrapgpu_buffer_t * buf_ = nullptr;
    scrapgpu_tensor_desc desc_{};
    scrapgpu_dtype dtype_ = SCRAPGPU_DTYPE_F32;
    scrapgpu_layout layout_ = SCRAPGPU_LAYOUT_NATURAL;
    int64_t in_features_ = 0;
    int64_t out_features_ = 0;
    std::string quant_type_;
};

// Registers the DTensor class and its ops (device_tensor.cpp); called from the TORCH_LIBRARY block after Weight.
void register_device_tensor(torch::Library & m);

} // namespace sgtorch
