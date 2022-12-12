#include "retinacuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda_types.hpp>

#include "RetinaCudaException.hpp"
#include "data/data.h"
#include "gpu/cuda_retina_kernels.cuh"
#include "layers/ConeLayer.h"
#include "layers/MGCellLayer.h"
#include "layers/PGCellLayer.h"
#include "simulations/ConeModel.h"
#include "simulations/PixelConeModel.h"
#include "utils/interp_utils.h"

#ifdef WITH_MATPLOTLIB
#include "matplotlib_cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

void throwRetinaCudaException(cudaError_t error) {
  if (error != cudaSuccess) {
    throw RetinaCudaException("RetinaCuda: Cuda error " + std::string(cudaGetErrorString(error)));
  }
}

RetinaCuda::RetinaCuda(int gpu) {
  gpuMCells = 0;
  gpuPCells = 0;

  int count;
  cudaError_t err = cudaGetDeviceCount(&count);

  if (count == 0 || err != cudaSuccess) {
    throw RetinaCudaException(std::string("No gpu avaliable: ") + cudaGetErrorString(err));
  } else if (count < gpu) {
    throw RetinaCudaException("Gpu not avaliable");
  }

  cv::cuda::setDevice(gpu);

  cv::cuda::DeviceInfo deviceInfos(gpu);

  std::cout << "Using gpu: " << deviceInfos.name() << std::endl;
}

RetinaCuda::~RetinaCuda() {}

void RetinaCuda::initRetina(Parameters param) {
  parameters = param;

  GCellsModel::GCellsDensityParams gc_density_params;
  gc_density_params.a = 0.9729;
  gc_density_params.r2 = 1.084;
  gc_density_params.re = 7.633;
  gc_density_params.max_cone_density = 14804;

  MGCellsSim::MGCellsDensityParams mgc_density_params;
  mgc_density_params.gc_params = gc_density_params;
  mgc_density_params.rm = 41.03;

  std::cout << "Initializing cone model ..." << std::endl;
  std::unique_ptr<ConeModel> cone_model_ptr = std::make_unique<ConeModel>(parameters.ph_config);
  std::cout << "Initializing pixel model ..." << std::endl;
  std::unique_ptr<PixelConeModel> pixel_model_ptr = std::make_unique<PixelConeModel>(parameters.pix_config);
  std::cout << "Initializing midget model ..." << std::endl;
  std::unique_ptr<MGCellsSim> mgc_model_ptr = std::make_unique<MGCellsSim>(mgc_density_params);
  std::cout << "Initializing parasol model ..." << std::endl;
  std::unique_ptr<PGCellsModel> pgc_model_ptr = std::make_unique<PGCellsModel>(mgc_density_params);

  std::cout << "Building cone layer ..." << std::endl;
  cone_layer_ptr_ = std::make_unique<ConeLayer>(cone_model_ptr, pixel_model_ptr, parameters.random_seed);
  std::cout << "Building midget layer ..." << std::endl;
  mgcells_layer_ptr_ = std::make_unique<MGCellLayer>(mgc_model_ptr, *cone_layer_ptr_, parameters.random_seed);
  std::cout << "Building parasol layer ..." << std::endl;
  pgcells_layer_ptr_ = std::make_unique<PGCellLayer>(pgc_model_ptr, *cone_layer_ptr_, parameters.random_seed);

  initPhotoGpu(cone_layer_ptr_->cones());
  initMCellsGpu(mgcells_layer_ptr_->mgcells());
  initPCellsGpu(pgcells_layer_ptr_->pgcells());

  initSelectiveCells();
}

#ifdef WITH_MATPLOTLIB
void RetinaCuda::plotLayersInfos() {
  if (cone_layer_ptr_) {
    cone_layer_ptr_->plotGraphs();
  }
  if (mgcells_layer_ptr_) {
    mgcells_layer_ptr_->plotGraphs();
  }
  if (pgcells_layer_ptr_) {
    pgcells_layer_ptr_->plotGraphs();
  }
  plt::show();
}
#endif

std::vector<Point> RetinaCuda::initSelectiveCells() {
  magnoMappingSrc.clear();
  magnoMappingDst.clear();
  const Cones &cones_cpu = cone_layer_ptr_->cones();

  unsigned int width = cones_cpu.width / 2;
  width -= width % BLOCK_SIZE;
  unsigned int height = cones_cpu.height / 2;
  height -= height % BLOCK_SIZE;
  for (unsigned int h = 0; h < height; h++) {
    // x = 0;
    for (unsigned int w = 0; w < width; w++) {
      magnoMappingSrc.push_back(Point(w * 2, h * 2));
      magnoMappingDst.push_back(Point(w, h));
    }
    // y++;
  }
  directive_width = width;
  directive_height = height;
  initDirectiveGpu(magnoMappingSrc, magnoMappingDst);

  return magnoMappingSrc;
}

cv::Mat RetinaCuda::drawConeMap() {
  const Cones &cones_cpu = cone_layer_ptr_->cones();
  cv::Mat output_mapping(cones_cpu.height, cones_cpu.width, CV_8UC3, cv::Vec3b(0, 0, 0));

  for (int y = 0; y < cones_cpu.height; y++) {
    for (int x = 0; x < cones_cpu.width; x++) {
      const auto &cone = cones_cpu.cones[x + y * cones_cpu.width];
      cv::Vec3b &color = output_mapping.at<cv::Vec3b>({x, y});
      switch (cone.type) {
        case PHOTO_TYPE::S_CONE:
          color[0] = 255;
          break;
        case PHOTO_TYPE::M_CONE:
          color[1] = 255;
          break;
        case PHOTO_TYPE::L_CONE:
          color[2] = 255;
          break;
        default:
          break;
      }
    }
  }
  return output_mapping;
}

void RetinaCuda::applyPhotoreceptorSampling(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst) {
  const Cones &cones_cpu = cone_layer_ptr_->cones();
  imgDst.create(cones_cpu.height, cones_cpu.width, CV_8UC1);
  imgDst.setTo(0);

  if (imgSrc.channels() == 1) {
    gpu::photoreceptorSampling1C(imgSrc, imgDst, gpuCones, cones_cpu.width, cones_cpu.height,
                                 cudaStreamDefault /*(cudaStream_t)cuda_stream*/);
    cudaError_t err = cudaGetLastError();
    throwRetinaCudaException(err);
  } else if (imgSrc.channels() == 3) {
    gpu::photoreceptorSampling3C(imgSrc, imgDst, gpuCones, cones_cpu.width, cones_cpu.height,
                                 cudaStreamDefault /*(cudaStream_t)cuda_stream*/);
    throwRetinaCudaException(cudaGetLastError());
  } else {
    std::cerr << "Not implemented" << std::endl;
  }
}

bool RetinaCuda::initMCellsGpu(const GanglionarCells &mgcells) {
  if (cudaMalloc((void **)&gpuMCells, sizeof(Ganglionar) * mgcells.gcells.size()) != cudaError_t::cudaSuccess) {
    return false;
  }
  cudaMemcpy(gpuMCells, mgcells.gcells.data(), sizeof(Ganglionar) * mgcells.gcells.size(), cudaMemcpyHostToDevice);
  return true;
}

bool RetinaCuda::initPCellsGpu(const GanglionarCells &pgcells) {
  if (cudaMalloc((void **)&gpuPCells, sizeof(Ganglionar) * pgcells.gcells.size()) != cudaError_t::cudaSuccess) {
    return false;
  }
  cudaMemcpy(gpuPCells, pgcells.gcells.data(), sizeof(Ganglionar) * pgcells.gcells.size(), cudaMemcpyHostToDevice);
  return true;
}

bool RetinaCuda::initDirectiveGpu(std::vector<Point> photoSrc, std::vector<Point> photoDst) {
  if (cudaMalloc((void **)&d_magnoMappingSrc, sizeof(Point) * photoSrc.size()) != cudaError_t::cudaSuccess) {
    return false;
  }
  cudaMemcpy(d_magnoMappingSrc, photoSrc.data(), sizeof(Point) * photoSrc.size(), cudaMemcpyHostToDevice);

  if (cudaMalloc((void **)&d_magnoMappingDst, sizeof(Point) * photoDst.size()) != cudaError_t::cudaSuccess) {
    return false;
  }
  cudaMemcpy(d_magnoMappingDst, photoDst.data(), sizeof(Point) * photoDst.size(), cudaMemcpyHostToDevice);

  this->magnoMappingSize = photoSrc.size();

  return true;
}

bool RetinaCuda::initPhotoGpu(const Cones &cones_cpu) {
  if (cudaMalloc((void **)&gpuCones, sizeof(Cone) * cones_cpu.cones.size()) != cudaError_t::cudaSuccess) {
    return false;
  }
  cudaMemcpy(gpuCones, cones_cpu.cones.data(), sizeof(Cone) * cones_cpu.cones.size(), cudaMemcpyHostToDevice);
  return true;
}

void RetinaCuda::applyParvoGC(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst) {
  if (imgDst.cols != mgcells_layer_ptr_->mgcells().width || imgDst.rows != mgcells_layer_ptr_->mgcells().height) {
    imgDst.create(mgcells_layer_ptr_->mgcells().height, mgcells_layer_ptr_->mgcells().width, CV_8UC1);
    imgDst.setTo(0);
  }

  gpu::multiConvolve(imgSrc, imgDst, gpuMCells, mgcells_layer_ptr_->mgcells().width,
                     mgcells_layer_ptr_->mgcells().height, cudaStreamDefault);
  throwRetinaCudaException(cudaGetLastError());
}

void RetinaCuda::applyMagnoGC(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst) {
  if (imgDst.cols != pgcells_layer_ptr_->pgcells().width || imgDst.rows != pgcells_layer_ptr_->pgcells().height) {
    imgDst.create(pgcells_layer_ptr_->pgcells().height, pgcells_layer_ptr_->pgcells().width, CV_8UC1);
    imgDst.setTo(0);
  }

  gpu::multiConvolve(imgSrc, imgDst, gpuPCells, pgcells_layer_ptr_->pgcells().width,
                     pgcells_layer_ptr_->pgcells().height, cudaStreamDefault);
  throwRetinaCudaException(cudaGetLastError());
}

void RetinaCuda::applyDirectiveGC(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst, cv::cuda::GpuMat &prevImage) {
  std::cout << "directive_height " << directive_height << "directive_width " << directive_width << std::endl;
  imgDst.create(directive_height, directive_width, CV_8UC1);
  imgDst.setTo(0);
  gpu::directionSelectiveComputation(imgSrc, imgDst, prevImage, d_magnoMappingSrc, d_magnoMappingDst, magnoMappingSize,
                                     cudaStreamDefault);
  throwRetinaCudaException(cudaGetLastError());
}
