#include "r32_transfer.h"

#include "opencl11/r32.h"
#include "tensor/tensor.h"

#include <algorithm>
#include <vector>

namespace scrapgpu_cuda {
namespace {

sgt::r32_geom geometry(WType t, int64_t K, int64_t rows) {
    const int64_t ne[4] = { K, rows, 1, 1 };
    return sgt::r32_geometry(sgt::make_contiguous(t == WType::Q4_0 ? SCRAPGPU_DTYPE_Q4_0 : SCRAPGPU_DTYPE_Q6_K, ne));
}

// the groups [g0, g1) of 32 rows are one chunk of about 24 MiB
int64_t chunk_groups(const sgt::r32_geom & G) {
    return std::max<int64_t>(1, (int64_t) ((size_t) 24 * 1024 * 1024 / (32 * G.rowb)));
}

bool transfer(Device & d, WType t, int64_t K, int64_t rows, uint8_t * nat, uint64_t dptr, CUstream s, bool to_dev) {
    const sgt::r32_geom G = geometry(t, K, rows);
    if (G.rowb == 0 || G.M <= 0 || G.nb <= 0 || (dptr & 15) != 0) {
        return false;
    }
    std::vector<uint8_t> abuf[3];
    const int64_t cg = chunk_groups(G);
    const int64_t ngrp = (rows + 31) / 32;
    for (int64_t gc = 0; gc < ngrp; gc += cg) {
        const int64_t rA = gc * 32;
        const int64_t rB = std::min<int64_t>(rows, std::min(ngrp, gc + cg) * 32);
        const int64_t nrows = rB - rA;
        uint8_t * ap[3] = { nullptr, nullptr, nullptr };
        size_t an[3] = { 0, 0, 0 };
        uint64_t ao[3] = { 0, 0, 0 };
        for (int k = 0; k < G.na; ++k) {
            an[k] = (size_t) nrows * (size_t) G.nb * (size_t) G.area[k].len;
            ao[k] = dptr + (uint64_t) G.M * (uint64_t) G.nb * sgt::r32_area_before(G, k) + (uint64_t) rA * (uint64_t) G.nb * (uint64_t) G.area[k].len;
            abuf[k].resize(an[k]);
            ap[k] = abuf[k].data();
        }
        uint8_t * hp = nat + (size_t) rA * G.rowb;
        if (to_dev) {
            cl11_rp_pack(G, hp, nrows, ap);
            for (int k = 0; k < G.na; ++k) {
                if (!d.h2d(ao[k], ap[k], an[k], s)) {
                    return false;
                }
            }
        } else {
            for (int k = 0; k < G.na; ++k) {
                if (!d.d2h(ap[k], ao[k], an[k], s)) {
                    return false;
                }
            }
            if (!d.stream_sync(s)) {
                return false;
            }
            cl11_rp_unpack(G, (const uint8_t * const *) ap, nrows, hp);
        }
    }
    // a copy from pageable memory has consumed the host buffer when the call returns; the device side is complete after the sync
    return d.stream_sync(s);
}

} // namespace

bool r32_upload(Device & d, WType t, int64_t K, int64_t rows, const void * nat, uint64_t dptr, CUstream s) {
    return transfer(d, t, K, rows, const_cast<uint8_t *>(static_cast<const uint8_t *>(nat)), dptr, s, true);
}

bool r32_download(Device & d, WType t, int64_t K, int64_t rows, void * nat, uint64_t dptr, CUstream s) {
    return transfer(d, t, K, rows, static_cast<uint8_t *>(nat), dptr, s, false);
}

} // namespace scrapgpu_cuda
