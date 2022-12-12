#pragma once

#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "gpu/retinastructs.h"
#include "simulations/ConeModelConfig.h"
#include "simulations/PixelConeModelConfig.h"
#include "utils/ramp_utils.h"

class ConeLayer;
class MGCellLayer;
class PGCellLayer;

/**
 * @brief Class simulating the retina response from a camera input
 *
 */
class RetinaCuda {
 public:
  struct Parameters {
    //! input image width/height in pixel
    int input_width = 0;
    int input_height = 0;

    //! Cone model configuration (ratio of each kind of cone)
    ConeModelConfig ph_config;
    //! Camera resolution and hfov
    PixelConeModelConfig pix_config;

    //! random seed to generate cone distribution
    int random_seed = 0;
  };

  /**
   * @brief Construct a new Retina Cuda object
   *
   * @param gpu gpu id to use
   */
  RetinaCuda(int gpu = 0);
  ~RetinaCuda();

  /**
   * @brief Initialize the retina model
   *
   * @param param
   */
  void initRetina(Parameters param);

  /**
   * @brief Generate the photoreceptor response from a given image
   *
   * @param imgSrc input grayscale or rgb image
   * @param imgDst photoreceptor response
   */
  void applyPhotoreceptorSampling(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst);

  /**
   * @brief Generate the Parvo ganglionar response from the given photoreceptor response
   *
   * @param imgSrc photoreceptor reponse
   * @param imgDst resulting parvo ganglionar response
   */
  void applyParvoGC(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst);

  /**
   * @brief Generate the Magno ganglionar response from the given photoreceptor response
   *
   * @param imgSrc photoreceptor reponse
   * @param imgDst resulting magno ganglionar response
   */
  void applyMagnoGC(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst);

  /**
   * @brief Generate the directive ganglionar response from the current and last photoreceptive response
   *
   * @param imgSrc photoreceptor reponse
   * @param imgDst directive ganglionar response
   * @param prevImage previous photoreceptor reponse
   */
  void applyDirectiveGC(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst, cv::cuda::GpuMat &prevImage);

  /**
   * @brief Generate an RGB image to visualize the cones distribution
   *
   * @return cv::Mat
   */
  cv::Mat drawConeMap();

#ifdef WITH_MATPLOTLIB
  void plotLayersInfos();
#endif

 protected:
  virtual std::vector<Point> initSelectiveCells();

 private:
  bool initPhotoGpu(const Cones &cones);
  bool initMCellsGpu(const GanglionarCells &mgcells);
  bool initPCellsGpu(const GanglionarCells &pcells);
  bool initDirectiveGpu(std::vector<Point> photoSrc, std::vector<Point> photoDst);

  Cone *gpuCones;
  Ganglionar *gpuMCells;
  Ganglionar *gpuPCells;

  Point *d_magnoMappingSrc;
  Point *d_magnoMappingDst;
  int directive_width;
  int directive_height;
  int magnoMappingSize;
  cv::cuda::GpuMat gpuChannelSampling;
  std::vector<Point> magnoMappingSrc;
  std::vector<Point> magnoMappingDst;
  Parameters parameters;

  std::unique_ptr<ConeLayer> cone_layer_ptr_;
  std::unique_ptr<MGCellLayer> mgcells_layer_ptr_;
  std::unique_ptr<PGCellLayer> pgcells_layer_ptr_;
};
