#include <builtin_types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <unistd.h>

#include "cuda_retina_kernels.cuh"
#include "cuda_runtime_api.h"

namespace gpu {

constexpr std::size_t PolarPixelMappingSize = 21;
constexpr float MaxPolarRadiusSquared = 2.0 * 2.0 + 1.0;
constexpr float ConeRadiusSquared = 0.5 * 0.5;

__constant__ int PolarPixelXMappingE[PolarPixelMappingSize] = {0,  1, 0, -1, 0,  1,  -1, -1, 1, 2, 0,
                                                               -2, 0, 2, 1,  -1, -2, -2, -1, 1, 2};
__constant__ int PolarPixelYMappingE[PolarPixelMappingSize] = {0, 0,  1, 0, -1, 1, 1, -1, -1, 0, 2,
                                                               0, -2, 1, 2, 2,  1, 1, -2, -2, -1};

Ganglionar* loadCellsArrayToGPU(Ganglionar* cellsArrayHost, int width, int height) {
  int size = width * height * sizeof(Ganglionar);
  Ganglionar* cellsArrayDevice;

  cudaMalloc((void**)&cellsArrayDevice, size);
  cudaMemcpy(cellsArrayDevice, cellsArrayHost, size, cudaMemcpyHostToDevice);

  return cellsArrayDevice;
}

void unloadArray(Ganglionar* cell) {
  // destruction des matrices, désormais inutilisées
  cudaFree(cell);
}

__global__ void photoreceptorSamplingKernel1C(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst,
                                              Cone* conesArray, int conesArrayWidth, int /*conesArrayHeight*/) {
  // Get our global thread ID
  int xdst = blockIdx.x * blockDim.x + threadIdx.x;
  int ydst = blockIdx.y * blockDim.y + threadIdx.y;

  Cone cone = conesArray[(xdst + ydst * conesArrayWidth)];

  if (cone.type == PHOTO_TYPE::NONE) {
    return;
  }
  int x = cone.center_x;
  int y = cone.center_y;

  //    if(xdst<imgDst.cols && ydst < imgDst.rows)
  imgDst(ydst, xdst) = imgSrc(y, x);
}

__global__ void photoreceptorSamplingKernel3C(cv::cuda::PtrStepSz<uchar3> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst,
                                              Cone* conesArray, int conesArrayWidth, int /*conesArrayHeight*/) {
  // Get our global thread ID
  int xdst = blockIdx.x * blockDim.x + threadIdx.x;
  int ydst = blockIdx.y * blockDim.y + threadIdx.y;

  Cone cone = conesArray[(xdst + ydst * conesArrayWidth)];

  if (cone.type == PHOTO_TYPE::NONE) {
    return;
  }
  int x = cone.center_x;
  int y = cone.center_y;

  uchar3 pixel = imgSrc(y, x);

  switch (cone.type) {
    case PHOTO_TYPE::S_CONE:
      imgDst(ydst, xdst) = pixel.x;
      break;
    case PHOTO_TYPE::M_CONE:
      imgDst(ydst, xdst) = pixel.y;
      break;
    case PHOTO_TYPE::L_CONE:
      imgDst(ydst, xdst) = pixel.z;
      break;
    default:
      break;
  }
}

__global__ void multiConvolveKernel(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst,
                                    Ganglionar* cellsArray, int cellsArrayWidth, int cellsArrayHeight) {
  // Get our global thread ID
  int xdst = blockIdx.x * blockDim.x + threadIdx.x;
  int ydst = blockIdx.y * blockDim.y + threadIdx.y;

  int nbcols = imgSrc.cols;
  int nbrows = imgSrc.rows;

  Ganglionar cell = cellsArray[(xdst + ydst * cellsArrayWidth)];

  if (cell.type == GC_RESPONSE_TYPE::NONE) {
    imgDst(ydst, xdst) = 0;
    return;
  }

  int x = cell.center_x;
  int y = cell.center_y;

  float in_radius_squarred = cell.intern_radius * cell.intern_radius;
  float ex_radius_squarred = cell.extern_radius * cell.extern_radius;

  int xi;
  int yi;

  int value_center = 0;
  int value_ext = 0;

  int nbCenter = 0;
  int nbOut = 0;

  if (ex_radius_squarred < ConeRadiusSquared) {
    if (cell.type == GC_RESPONSE_TYPE::ON) {
      value_center += imgSrc(y, x);
    } else {
      value_center -= imgSrc(y, x);
    }
    nbCenter = 1;
    imgDst(ydst, xdst) = 10;
    return;
  } else if (ex_radius_squarred < MaxPolarRadiusSquared) {
    // If the kernel are too small, use polar loop
    int inside_cones = in_radius_squarred / ConeRadiusSquared;
    int all_cones = ex_radius_squarred / ConeRadiusSquared;

    int cone_index = 0;
    while (cone_index < inside_cones) {
      if (cell.type == GC_RESPONSE_TYPE::ON) {
        value_center += imgSrc(y + PolarPixelYMappingE[cone_index], x + PolarPixelXMappingE[cone_index]);
      } else {
        value_center -= imgSrc(y + PolarPixelYMappingE[cone_index], x + PolarPixelXMappingE[cone_index]);
      }
      nbCenter++;
      cone_index++;
    };

    while (cone_index < all_cones) {
      if (cell.type == GC_RESPONSE_TYPE::ON) {
        value_ext -= imgSrc(y + PolarPixelYMappingE[cone_index], x + PolarPixelXMappingE[cone_index]);
      } else {
        value_ext += imgSrc(y + PolarPixelYMappingE[cone_index], x + PolarPixelXMappingE[cone_index]);
      }
      nbOut++;
      cone_index++;
    };

  } else {
    for (xi = -cell.extern_radius; xi <= cell.extern_radius; xi++) {
      for (yi = -cell.extern_radius; yi <= cell.extern_radius; yi++) {
        if (x + xi > 0 && x + xi < nbcols && y + yi > 0 && y + yi < nbrows) {
          if (xi * xi + yi * yi <= in_radius_squarred) {   // if we are in the radius
            if (cell.type == GC_RESPONSE_TYPE::ON) {
              value_center += imgSrc(y + yi, x + xi);
            } else {
              value_center -= imgSrc(y + yi, x + xi);
            }
            nbCenter++;
          } else if (xi * xi + yi * yi <= ex_radius_squarred) {
            if (cell.type == GC_RESPONSE_TYPE::ON) {
              value_ext -= imgSrc(y + yi, x + xi);
            } else {
              value_ext += imgSrc(y + yi, x + xi);
            }
            nbOut++;
          }
        } else {
          // receptive field outside cone map
          imgDst(ydst, xdst) = 255;
          return;
        }
      }
    }
  }

  int total_value;

  // the ganglionar response is centred on 128
  // [0,128[ low pulsing frequencies
  // ]128,255] high pulsing frequencies
  if (nbOut == 0) {
    total_value = value_center / (float)nbCenter / 2.0 + 128;
  } else if (nbCenter == 0) {
    total_value = 128;
  } else {
    total_value = (value_center / (float)nbCenter + value_ext / (float)nbOut) / 2.0 + 128;   //*cell.extern_radius;
    // total_value =  (value_center/(float)nbCenter + value_ext/(float)nbOut)/2.0;//*cell.extern_radius;
  }

  if (total_value < 0) {
    total_value = 0;
  } else if (total_value > 255) {
    total_value = 255;
  }

  if (xdst < imgDst.cols && ydst < imgDst.rows)
    imgDst(ydst, xdst) = total_value;
}

__global__ void legacyMultiConvolveKernel(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst,
                                          Ganglionar* cellsArray, int cellsArrayWidth, int cellsArrayHeight) {
  // Get our global thread ID
  int xdst = blockIdx.x * blockDim.x + threadIdx.x;
  int ydst = blockIdx.y * blockDim.y + threadIdx.y;

  int nbcols = imgSrc.cols;
  int nbrows = imgSrc.rows;

  Ganglionar cell = cellsArray[(xdst + ydst * cellsArrayWidth)];

  if (cell.type == GC_RESPONSE_TYPE::NONE) {
    imgDst(ydst, xdst) = 0;
  }

  int x = cell.center_x;
  int y = cell.center_y;

  int in_radius_squarred = cell.intern_radius * cell.intern_radius;
  int ex_radius_squarred = cell.extern_radius * cell.extern_radius;

  int xi;
  int yi;

  int value_center = 0;
  int value_ext = 0;

  int nbCenter = 0;
  int nbOut = 0;

  if (ex_radius_squarred == 1) {
    if (cell.type == GC_RESPONSE_TYPE::ON) {
      value_center += (imgSrc(y, x) - 128);
    } else {
      value_center -= (imgSrc(y, x) - 128);
    }
    nbCenter = 1;
  } else {
    for (xi = -cell.extern_radius; xi <= cell.extern_radius; xi++) {
      for (yi = -cell.extern_radius; yi <= cell.extern_radius; yi++) {
        if (x + xi > 0 && x + xi < nbcols && y + yi > 0 && y + yi < nbrows) {
          if (xi * xi + yi * yi < in_radius_squarred) {   // if we are in the radius
            if (cell.type == GC_RESPONSE_TYPE::ON) {
              value_center += (imgSrc(y + yi, x + xi) - 128);
            } else {
              value_center -= (imgSrc(y + yi, x + xi) - 128);
            }
            nbCenter++;
          } else if (xi * xi + yi * yi < ex_radius_squarred) {
            if (cell.type == GC_RESPONSE_TYPE::ON) {
              value_ext -= (imgSrc(y + yi, x + xi) - 128);
            } else {
              value_ext += (imgSrc(y + yi, x + xi) - 128);
            }
            nbOut++;
          }
        }
      }
    }
  }

  int total_value;

  if (nbOut == 0) {
    nbOut = 1;
  }
  if (nbCenter == 0) {
    total_value = 128;
  } else {
    total_value = 128 + (value_center / (float)nbCenter + value_ext / (float)nbOut) / 2.0;   //*cell.extern_radius;
  }

  if (total_value < 0) {
    total_value = 0;
  } else if (total_value > 255) {
    total_value = 255;
  }

  if (xdst < imgDst.cols && ydst < imgDst.rows)
    imgDst(ydst, xdst) = total_value;
}

__global__ void directionSelectiveKernel(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst,
                                         cv::cuda::PtrStepSz<u_char> imgPrev, Point* directiveMappingSrc,
                                         Point* directiveMappingDst, int size) {
  // Get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id >= size) {
    return;
  }
  Point pointSrc = directiveMappingSrc[id];
  Point pointDst = directiveMappingDst[id];
  int xdst = pointSrc.x;
  int ydst = pointSrc.y;

  int nbcols = imgSrc.cols;
  int nbrows = imgSrc.rows;

  int type = xdst % 4;   // There are 4 types, to top, left right bottom

  int response = 0;
  int delta = 0;
  if (xdst % 2 == 0) {
    delta += 1;
  } else {
    delta -= 1;
  }
  int dx = xdst + delta;
  int dy = ydst + delta;

  if (type < 2 && dx < nbcols && dx >= 0) {
    response = ((int)imgPrev(ydst, dx) + (int)imgSrc(ydst, xdst)) -
               ((int)imgSrc(ydst, dx) + (int)imgPrev(ydst, xdst));   // on directive
  } else if (dy < nbrows && dy >= 0) {
    response = ((int)imgPrev(dy, xdst) + (int)imgSrc(ydst, xdst)) -
               ((int)imgSrc(dy, xdst) + (int)imgPrev(ydst, xdst));   // on directive
  }

  if (response < 0) {
    response = 0;
  } else if (response > 255) {
    response = 255;
  }

  imgDst(pointDst.y, pointDst.x) = response;
}

void photoreceptorSampling1C(cv::cuda::PtrStepSz<uchar> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Cone* coneArrayGPU,
                             int conesWidth, int conesHeight, cudaStream_t stream) {
  dim3 grid, block;

  // Number of threads in each thread block
  block.x = BLOCK_SIZE;
  block.y = BLOCK_SIZE;

  grid.x = (int)ceil((float)(imgDst.cols) / block.x);
  grid.y = (int)ceil((float)(imgDst.rows) / block.y);
  photoreceptorSamplingKernel1C<<<grid, block>>>(imgSrc, imgDst, coneArrayGPU, conesWidth, conesHeight);
}

void photoreceptorSampling3C(cv::cuda::PtrStepSz<uchar3> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Cone* coneArrayGPU,
                             int conesWidth, int conesHeight, cudaStream_t stream) {
  dim3 grid, block;

  // Number of threads in each thread block
  block.x = BLOCK_SIZE;
  block.y = BLOCK_SIZE;

  grid.x = (int)ceil((float)(imgDst.cols) / block.x);
  grid.y = (int)ceil((float)(imgDst.rows) / block.y);
  photoreceptorSamplingKernel3C<<<grid, block>>>(imgSrc, imgDst, coneArrayGPU, conesWidth, conesHeight);
}

void multiConvolve(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Ganglionar* cellsArrayGPU,
                   int cellsArrayWidth, int cellsArrayHeight, cudaStream_t stream) {
  dim3 grid, block;

  // Number of threads in each thread block
  block.x = BLOCK_SIZE;
  block.y = BLOCK_SIZE;

  grid.x = (int)ceil((float)(imgDst.cols) / block.x);
  grid.y = (int)ceil((float)(imgDst.rows) / block.y);

  multiConvolveKernel<<<grid, block>>>(imgSrc, imgDst, cellsArrayGPU, cellsArrayWidth, cellsArrayHeight);
}

void directionSelectiveComputation(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst,
                                   cv::cuda::PtrStepSz<u_char> imgPrev, Point* directiveMappingSrc,
                                   Point* directiveMappingDst, int directiveMappingSize, cudaStream_t stream) {
  dim3 grid, block;

  // Number of threads in each thread block
  block.x = 256;

  grid.x = (int)ceil((float)(directiveMappingSize) / (float)block.x);
  directionSelectiveKernel<<<grid, block>>>(imgSrc, imgDst, imgPrev, directiveMappingSrc, directiveMappingDst,
                                            directiveMappingSize);
}

}   // namespace gpu
