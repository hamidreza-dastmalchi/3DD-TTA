/**********************************
 * Original Author: Haoqiang Fan
 * Modified by: Kaichun Mo
 * Updated for ATen API by: [Your Name]
 *********************************/

#ifndef _EMD_KERNEL
#define _EMD_KERNEL

#include <cmath>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>  // For AT_CUDA_CHECK and CUDA utilities
#include <c10/cuda/CUDAStream.h>    // For managing CUDA streams

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/********************************
* Forward kernel for approxmatch
*********************************/

template <typename scalar_t>
__global__ void approxmatch(int b, int n, int m, const scalar_t* __restrict__ xyz1,
                            const scalar_t* __restrict__ xyz2, scalar_t* __restrict__ match,
                            scalar_t* temp) {
    scalar_t* remainL = temp + blockIdx.x * (n + m) * 2;
    scalar_t* remainR = temp + blockIdx.x * (n + m) * 2 + n;
    scalar_t* ratioL = temp + blockIdx.x * (n + m) * 2 + n + m;
    scalar_t* ratioR = temp + blockIdx.x * (n + m) * 2 + n + m + n;
    scalar_t multiL, multiR;

    if (n >= m) {
        multiL = 1;
        multiR = n / m;
    } else {
        multiL = m / n;
        multiR = 1;
    }

    const int Block = 1024;
    __shared__ scalar_t buf[Block * 4];

    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        for (int j = threadIdx.x; j < n * m; j += blockDim.x)
            match[i * n * m + j] = 0;
        for (int j = threadIdx.x; j < n; j += blockDim.x)
            remainL[j] = multiL;
        for (int j = threadIdx.x; j < m; j += blockDim.x)
            remainR[j] = multiR;
        __syncthreads();

        for (int j = 7; j >= -2; j--) {
            scalar_t level = -powf(4.0f, j);
            if (j == -2) {
                level = 0;
            }
            for (int k0 = 0; k0 < n; k0 += blockDim.x) {
                int k = k0 + threadIdx.x;
                scalar_t x1 = 0, y1 = 0, z1 = 0;
                if (k < n) {
                    x1 = xyz1[i * n * 3 + k * 3 + 0];
                    y1 = xyz1[i * n * 3 + k * 3 + 1];
                    z1 = xyz1[i * n * 3 + k * 3 + 2];
                }
                scalar_t suml = 1e-9f;
                for (int l0 = 0; l0 < m; l0 += Block) {
                    int lend = min(m, l0 + Block) - l0;
                    for (int l = threadIdx.x; l < lend; l += blockDim.x) {
                        scalar_t x2 = xyz2[i * m * 3 + l0 * 3 + l * 3 + 0];
                        scalar_t y2 = xyz2[i * m * 3 + l0 * 3 + l * 3 + 1];
                        scalar_t z2 = xyz2[i * m * 3 + l0 * 3 + l * 3 + 2];
                        buf[l * 4 + 0] = x2;
                        buf[l * 4 + 1] = y2;
                        buf[l * 4 + 2] = z2;
                        buf[l * 4 + 3] = remainR[l0 + l];
                    }
                    __syncthreads();
                    for (int l = 0; l < lend; l++) {
                        scalar_t x2 = buf[l * 4 + 0];
                        scalar_t y2 = buf[l * 4 + 1];
                        scalar_t z2 = buf[l * 4 + 2];
                        scalar_t d = level * ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1));
                        scalar_t w = __expf(d) * buf[l * 4 + 3];
                        suml += w;
                    }
                    __syncthreads();
                }
                if (k < n)
                    ratioL[k] = remainL[k] / suml;
            }
            __syncthreads();
        }
    }
}

/* ApproxMatch forward interface */
at::Tensor ApproxMatchForward(const at::Tensor xyz1, const at::Tensor xyz2) {
    const auto b = xyz1.size(0);
    const auto n = xyz1.size(1);
    const auto m = xyz2.size(1);

    TORCH_CHECK(xyz2.size(0) == b, "Batch size mismatch between xyz1 and xyz2");
    TORCH_CHECK(xyz1.size(2) == 3 && xyz2.size(2) == 3, "Input tensors must have size 3 in the last dimension");
    CHECK_INPUT(xyz1);
    CHECK_INPUT(xyz2);

    auto match = at::zeros({b, m, n}, xyz1.options());
    auto temp = at::zeros({b, (n + m) * 2}, xyz1.options());

    AT_DISPATCH_FLOATING_TYPES(xyz1.scalar_type(), "ApproxMatchForward", ([&] {
        approxmatch<scalar_t><<<32, 512>>>(
            b, n, m, 
            xyz1.data_ptr<scalar_t>(), 
            xyz2.data_ptr<scalar_t>(), 
            match.data_ptr<scalar_t>(), 
            temp.data_ptr<scalar_t>());
    }));
    AT_CUDA_CHECK(cudaGetLastError());

    return match;
}

#endif
