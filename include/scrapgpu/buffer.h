#ifndef SCRAPGPU_BUFFER_H
#define SCRAPGPU_BUFFER_H

#include "scrapgpu/device.h"

#ifdef __cplusplus
extern "C" {
#endif

SCRAPGPU_API scrapgpu_status scrapgpu_buffer_alloc(scrapgpu_device_t * device, size_t size, scrapgpu_buffer_t ** out_buffer);
SCRAPGPU_API scrapgpu_status scrapgpu_buffer_free(scrapgpu_buffer_t * buffer);
SCRAPGPU_API scrapgpu_status scrapgpu_buffer_write(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, const void * data, size_t size, int blocking);
SCRAPGPU_API scrapgpu_status scrapgpu_buffer_read(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, void * data, size_t size, int blocking);
SCRAPGPU_API scrapgpu_status scrapgpu_buffer_copy(scrapgpu_stream_t * stream, scrapgpu_buffer_t * dst, uint64_t dst_offset, scrapgpu_buffer_t * src, uint64_t src_offset, size_t size, int blocking);
SCRAPGPU_API scrapgpu_status scrapgpu_buffer_fill(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, size_t size, uint8_t byte, int blocking);

// Process-wide counters of the host <-> device transfers requested through the public API
// (scrapgpu_buffer_write/read, scrapgpu_tensor_upload/download), for tests and tuning that must show a
// device-resident pipeline moves data only at its boundaries. Device-to-device copies are not counted.
typedef struct scrapgpu_transfer_stats {
    size_t struct_size;
    uint64_t host_to_device_calls;
    uint64_t host_to_device_bytes;
    uint64_t device_to_host_calls;
    uint64_t device_to_host_bytes;
} scrapgpu_transfer_stats;

SCRAPGPU_API scrapgpu_status scrapgpu_transfer_stats_get(scrapgpu_transfer_stats * out);
SCRAPGPU_API void scrapgpu_transfer_stats_reset(void);

#ifdef __cplusplus
}
#endif

#endif
