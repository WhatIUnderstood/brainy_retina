#ifndef CUDA_BRAIN_KERNELS_H
#define CUDA_BRAIN_KERNELS_H

#include <builtin_types.h>
#include <stdint.h>

#include "retinastructs.h"
#include <opencv/cv.h>
#include "opencv2/cuda.hpp"
#include "Cuda/cuda_image.h"

namespace gpu{

//typedef const unsigned int cuint;

void channelSampling(cv::cuda::PtrStepSz<uchar3> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, cv::cuda::PtrStepSz<u_char> samplingMapGPU,cudaStream_t stream);
void multiConvolve(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Cell* cellsArrayGPU, int cellsArrayWidth, int cellsArrayHeight,cudaStream_t stream);

}
#endif // CUDA_BRAIN_H

