#pragma once

#include <builtin_types.h>
#include <stdint.h>

#include "opencv2/core/cuda.hpp"
#include "retinastructs.h"

namespace gpu {
/**
 * @brief Convert a 3 channels image into a 1 channel image by selecting, for each pixel, R or G or
 * B value. The channel selection is made according to the given samplingMapGPU map
 *
 * @param imgSrc input 3 channels image to process.
 * @param imgDst resulting image
 * @param samplingMapGPU 2D array with the same size of imgDst. It can have only 3 values: 0, 1
 * or 2. These three value correspond to B, G, R channels. It determines which chanel should be
 * picked to build imgDsr
 * @param stream cuda stream
 */
void channelSampling(cv::cuda::PtrStepSz<uchar3> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst,
                     cv::cuda::PtrStepSz<u_char> samplingMapGPU, cudaStream_t stream);

/**
 * @brief Simulate cones photoreceptive response to a grayscale image.
 *
 * @param imgSrc input grayscale image to process
 * @param imgDst resulting photoreceptive image
 * @param coneArrayGPU retina cones to consider
 * @param conesWidth width of the coneArrayGPU array
 * @param conesHeight height of the coneArrayGPU array
 * @param stream cuda stream
 */
void photoreceptorSampling1C(cv::cuda::PtrStepSz<uchar> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Cone *coneArrayGPU,
                             int conesWidth, int conesHeight, cudaStream_t stream);

/**
 * @brief Simulate cones photoreceptive response to a rgb image.
 *
 * @param imgSrc input rgb image to process
 * @param imgDst resulting photoreceptive image
 * @param coneArrayGPU retina cones to consider
 * @param conesWidth width of the coneArrayGPU array
 * @param conesHeight height of the coneArrayGPU array
 * @param stream cuda stream
 */
void photoreceptorSampling3C(cv::cuda::PtrStepSz<uchar3> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Cone *coneArrayGPU,
                             int conesWidth, int conesHeight, cudaStream_t stream);

/**
 * @brief Simulate the Ganglionar Cells response from a photoreceptive response
 *
 * @param imgSrc photoreceptive input image
 * @param imgDst resulting ganglionar cells responses
 * @param cellsArrayGPU ganglionar cells array
 * @param cellsArrayWidth width of the ganglionar array
 * @param cellsArrayHeight height of the ganglionar array
 * @param stream cuda stream
 */
void multiConvolve(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Ganglionar *cellsArrayGPU,
                   int cellsArrayWidth, int cellsArrayHeight, cudaStream_t stream);

/**
 * @brief Simulate the Directive selective response from a photoreceptive response
 *
 * @param imgSrc photoreceptive input image
 * @param imgDst resulting directive cells responses
 * @param imgPrev previous photoreceptive image
 * @param directiveMappingSrc current directive cells status
 * @param directiveMappingDst resulting directive cells status
 * @param directiveMappingSize directive cells size
 * @param stream cuda stream
 */
void directionSelectiveComputation(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst,
                                   cv::cuda::PtrStepSz<u_char> imgPrev, Point *directiveMappingSrc,
                                   Point *directiveMappingDst, int directiveMappingSize, cudaStream_t stream);

}   // namespace gpu
