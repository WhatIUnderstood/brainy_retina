#ifndef CUDA_BRAIN_KERNELS_H
#define CUDA_BRAIN_KERNELS_H

#include <builtin_types.h>
#include <stdint.h>

#include "retinastructs.h"
#include "opencv2/core/cuda.hpp"

namespace gpu
{
void channelSampling(cv::cuda::PtrStepSz<uchar3> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, cv::cuda::PtrStepSz<u_char> samplingMapGPU, cudaStream_t stream);
void photoreceptorSampling(cv::cuda::PtrStepSz<uchar3> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Cone *coneArrayGPU, int conesWidth, int conesHeight, cudaStream_t stream);
void photoreceptorSampling1C(cv::cuda::PtrStepSz<uchar> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Cone *coneArrayGPU, int conesWidth, int conesHeight, cudaStream_t stream);
void photoreceptorSampling3C(cv::cuda::PtrStepSz<uchar3> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Cone *coneArrayGPU, int conesWidth, int conesHeight, cudaStream_t stream);
void multiConvolve(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Ganglionar *cellsArrayGPU, int cellsArrayWidth, int cellsArrayHeight, cudaStream_t stream);
void directionSelectiveComputation(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, cv::cuda::PtrStepSz<u_char> imgPrev, Point *directiveMappingSrc, Point *directiveMappingDst, int directiveMappingSize, cudaStream_t stream);

//void sparse(cv::cuda::PtrStepSz<u_char> imgSrc, int depth, GpuBitArray2D &imgDst, unsigned char minval, unsigned char maxVal, cudaStream_t stream);

void discretise(cv::cuda::PtrStepSz<u_char> imgSrc, int depth, cv::cuda::PtrStepSz<u_char> imgDst, unsigned char minval, unsigned char maxVal, cudaStream_t stream);
} // namespace gpu
#endif // CUDA_BRAIN_H
