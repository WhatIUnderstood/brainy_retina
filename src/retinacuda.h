#pragma once

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/core/cuda.hpp"
#include "Cuda/retinastructs.h"
#include <random>
#include <iostream>
#include "Cuda/cuda_arrays.h"

#include "Utils/ramp_utils.h"

#include "simulations/ConeModel.h"
#include "simulations/PixelConeModel.h"
#include "simulations/MGCellsSim.h"
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

    //
    void sparse(cv::cuda::GpuMat &imgSrc, int bits, GpuBitArray2D &output, unsigned char min_value = 0, unsigned char max_value = 255);

    // Utils
    cv::Mat drawConeMap();

    //Test
    void discretise(cv::cuda::GpuMat &imgSrc, int vals, cv::cuda::GpuMat &output, unsigned char min_value = 0, unsigned char max_value = 255);
    void addKernels();

protected:
    virtual void initCone(int inputWidth, int inputHeight);
    virtual std::vector<Ganglionar> initGanglionarCells(int conesWidth, int conesHeight);
    virtual std::vector<Point> initSelectiveCells();
    // virtual cv::Mat initConeSampling(int cellsArrayWidth, int cellsArrayHeight);

private:
    enum class GC_TYPE
    {
        Midget,
        Parasol
    };

private:
    bool initPhotoGpu();
    bool initCellsGpu(std::vector<Ganglionar> cellsArrayCPU, int cellsArrayWidth, int cellsArrayHeight);
    bool initDirectiveGpu(std::vector<Point> photoSrc, std::vector<Point> photoDst);

    /// Generation functions ///
    double setRandomSeed(int val);
    double getRandom();

    template <typename OUTPUT_TYPE>
    OUTPUT_TYPE weightedRandom(std::vector<std::pair<float, OUTPUT_TYPE>> probabilities)
    {
        float proba_sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0, [](float sum, const std::pair<float, OUTPUT_TYPE> &pair) {
            return sum + pair.first;
        });
        if (proba_sum > 1)
        {
            throw std::invalid_argument("weightedRandom cannot have probabilities sum above 1");
        }
        // sort the probabilities
        std::sort(probabilities.begin(), probabilities.end(), [](const std::pair<float, OUTPUT_TYPE> &a, const std::pair<float, OUTPUT_TYPE> &b) {
            return a.first < b.first;
        });

        float random = getRandom();
        for (std::pair<float, OUTPUT_TYPE> proba : probabilities)
        {
            if (random < proba.first)
            {
                return proba.second;
            }
            else
            {
                random -= proba.first;
            }
        }
    }

    class LINEAR_RAMP
    {
        double initial_value;
        double final_value;
        double transitional_start;
        double transitional_stop;
    };

    /**
     * @brief
     *
     * @param distance_from_center
     * @return double
     */
    double mgc_dentric_coverage(float distance_from_center)
    {
        return linearRamp(distance_from_center, midget_gc_field_ramp_);
    }

    /**
     * @brief Magno ganglionar cells position mapping with cones
     * _____         ______
     *      \       /
     *       \     /
     *        -----
     *        fovea
     *  <- gc_fovea_radius ->
     * @param distance_from_center
     * @return double
     */
    double mgc_position_mapping(double distance_from_center)
    {
        return linearRampIntegral(distance_from_center, midget_gc_ramp_);
    }

    /**
     * @brief Convert a cone distance from the center to a pixel distance.
     * Pixels have the same density on all the image unlike the cones.
     *      /\
     * ____/  \___
     *
     * @param cone_distance_from_center
     * @return double
     */
    double cone_distance_mapping(double cone_distance_from_center)
    {
        return linearRampIntegral(cone_distance_from_center, pixel_per_cone_ramp_);
    }

    double getDistanceFromCenter(double pix_x, double pix_y, double width, double height)
    {
        return sqrt(std::pow(pix_x - width / 2.0, 2) + std::pow(pix_y - height / 2.0, 2));
    }

    //    double getConesMaxRadius(){
    //        return sqrt((double)parameters*parameters.cones_width + parameters.cones_height*parameters.cones_height)/2;
    //    }

    cv::Vec2f getDirectionFromCenter(cv::Point topLeftPosition, cv::Size size)
    {
        return cv::normalize(cv::Vec2f(topLeftPosition.x - size.width / 2.0, topLeftPosition.y - size.height / 2.0));
    }

    cv::Point getPosition(double a_distance_from_center, cv::Size a_size, cv::Vec2f direction)
    {
        cv::Vec2f normalizedDirection = cv::normalize(direction);
        double x_center = a_distance_from_center * normalizedDirection[0];
        double y_center = a_distance_from_center * normalizedDirection[1];

        return cv::Point(round(x_center + a_size.width / 2.0), round(y_center + a_size.height / 2.0));
    }

    //    cv::Point getPosition(double src_distance, int magno_x, int magno_y){
    //        double magno_center_x = magno_x-cellsArrayWidth/2.0;
    //        double magno_center_y = magno_y-cellsArrayHeight/2.0;
    //        double magno_distance = sqrt(magno_center_x*magno_center_x+magno_center_y*magno_center_y);
    //        double x_center = src_distance*magno_center_x/magno_distance;
    //        double y_center = src_distance*magno_center_y/magno_distance;
    //        return cv::Point(x_center+parameters.input_width/2.0,y_center+parameters.input_height/2.0);
    //    }

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
    int cellsArrayWidth;
    int cellsArrayHeight;
    // int conesArrayWidth;
    // int conesArrayHeight;
    Parameters parameters;
    //std::vector<int> coneMarge;
    //std::shared_ptr<void *> cuda_stream_ptr;

    ramp_utils::RampParameters midget_gc_field_ramp_;

    ramp_utils::RampParameters pixel_per_cone_ramp_;
    ramp_utils::RampParameters midget_gc_ramp_;
    ramp_utils::RampParameters parasol_gc_ramp_;

    std::mt19937 mt_rand;

    Cones cones_cpu_;

    std::unique_ptr<ConeModel> cone_model_ptr_;
    std::unique_ptr<PixelConeModel> photo_sim_ptr_;
};
