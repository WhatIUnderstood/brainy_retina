#pragma once

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/core/cuda.hpp"
#include "Cuda/retinastructs.h"
#include <random>
#include <iostream>

#include "Utils/ramp_utils.h"

#include "simulations/ConeModel.h"
#include "simulations/PixelConeModel.h"
#include "simulations/MGCellsSim.h"

class ConeLayer;
class MGCellLayer;

/**
 * @brief Class performing retina processing from a camera input
 *
 */
class RetinaCuda
{
public:
    struct Parameters
    {
        ///// Raw input params /////
        int input_width = 0;  // in pixels
        int input_height = 0; // in pixels

        ConeModelConfig ph_config;
        PixelConeModelConfig pix_config;

        //Photoreceptor params
        double ph_fovea_radius; //radius in pixels that are inside the fovea. 60M cones in human fovea
        double ph_fovea_pixels_by_cone;
        double ph_max_pixels_by_cone;
        double ph_max_pixels_by_cone_radius;
        float ph_S_cone_ratio = 0;
        float ph_M_cone_ratio = 0;
        float ph_L_cone_ratio = 0;

        //Ganglionar params
        double gc_fovea_radius;
        double gc_fovea_cones_by_cell;
        double gc_midget_fovea_cones_by_cell;
        double gc_parasol_fovea_cones_by_cell;
        double gc_fovea_inter_cells_distance;
        double gc_max_cones_by_cell;
        double gc_max_cones_by_cell_radius;
        double gc_max_inter_cells_distance;

        int random_seed = 0;
    };

    RetinaCuda(int gpu = 0);
    ~RetinaCuda();
    void initRetina(Parameters param);

    ///
    /// \brief applyPhotoreceptorSampling
    /// \param imgSrc
    /// \param imgDst
    ///
    void applyPhotoreceptorSampling(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst);

    ///
    /// \brief
    /// \param imgSrc
    /// \param imgDst
    ///
    void applyParvoGC(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst);

    ///
    /// \brief
    /// \param imgSrc
    /// \param imgDst
    /// \param prevImage
    ///
    void applyDirectiveGC(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst, cv::cuda::GpuMat &prevImage);

    // Utils
    cv::Mat drawConeMap();

    //Test
    void discretise(cv::cuda::GpuMat &imgSrc, int vals, cv::cuda::GpuMat &output, unsigned char min_value = 0, unsigned char max_value = 255);

protected:
    virtual std::vector<Point> initSelectiveCells();

private:
    enum class GC_TYPE
    {
        Midget,
        Parasol
    };

private:
    bool initPhotoGpu(const Cones &cones);
    bool initCellsGpu(const GanglionarCells &mgcells);
    bool initDirectiveGpu(std::vector<Point> photoSrc, std::vector<Point> photoDst);

    Cone *gpuCones;
    Ganglionar *gpuCells;
    Point *d_magnoMappingSrc;
    Point *d_magnoMappingDst;
    int directive_width;
    int directive_height;
    int magnoMappingSize;
    cv::cuda::GpuMat gpuChannelSampling;
    std::vector<Point> magnoMappingSrc;
    std::vector<Point> magnoMappingDst;
    Parameters parameters;

    ramp_utils::RampParameters midget_gc_field_ramp_;

    ramp_utils::RampParameters pixel_per_cone_ramp_;
    ramp_utils::RampParameters midget_gc_ramp_;
    ramp_utils::RampParameters parasol_gc_ramp_;

    std::unique_ptr<ConeLayer> cone_layer_ptr_;
    std::unique_ptr<MGCellLayer> mgcells_layer_ptr_;
};
