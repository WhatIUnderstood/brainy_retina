#pragma once
#include "../Cuda/retinastructs.h"
#include "../Utils/interp_utils.h"
#include "../data/data.h"

#include <opencv2/core.hpp>
#include <stdlib.h>
#include <math.h>
#include <memory>

#include "Utils/convertion.h"
#include "ConeModel.h"

struct PixelConeModelConfig
{
    double camera_hfov = 0; //degree
    int camera_width = 0;
    int camera_height = 0;
};

class PixelConeModel
{
public:
    PixelConeModel(const PixelConeModelConfig &config, const ConeModel &cone_model)
    {
        config_ = config;
        cone_model_ptr_ = std::make_unique<ConeModel>(cone_model);

        f_pix_ = config_.camera_width / (2 * std::tan(config_.camera_hfov * Deg_to_rad / 2.0));

        // Use pinhole model like
        pixel_densities_deg2_ = extractPixelDensityByDeg2();
        pixels_linear_density_integral_ = interp_utils::computeLinearDensityIntegral<double>(pixel_densities_deg2_);
    }

    int getHeight()
    {
        return config_.camera_height;
    }

    int getWidth()
    {
        return config_.camera_width;
    }

    int getMaxRadius()
    {
        return static_cast<int>(config_.camera_hfov / 2.0);
    }

    double getMaxEccentricity()
    {
        return config_.camera_hfov / 2.0;
    }

    double getMaxVEccentricity()
    {
        return config_.camera_hfov / (double)config_.camera_width * (double)config_.camera_height;
    }

    int getSimulatedConeRadius()
    {
        return static_cast<int>(cone_model_ptr_->getConeIndex(config_.camera_hfov / 2.0));
    }

    double getConeIndex(double ecc_deg)
    {
        return cone_model_ptr_->getConeIndex(ecc_deg);
    }

    /**
     * @brief Get the Pixel Index object.
     *
     * @param cone_index
     * @return double
     */
    double getPixelIndex(int cone_index)
    {
        double cone_angular_pose = cone_model_ptr_->getConeAngularPose(cone_index);
        return interp_utils::lin_interp(cone_angular_pose, pixels_linear_density_integral_.x, pixels_linear_density_integral_.y, -1.0, -1.0);
    }

    double
    getPixelDensityDeg2(double ecc_deg)
    {
        if (std::abs(ecc_deg) > config_.camera_hfov / 2.0)
        {
            return 0;
        }
        return std::pow((f_pix_ * (std::tan((ecc_deg + 0.5) * Deg_to_rad) - std::tan((ecc_deg - 0.5) * Deg_to_rad))), 2);
    }

    Graph<double> extractPixelDensityByDeg2()
    {
        Graph<double> pixel_density_deg2;
        for (double i = 0; i < config_.camera_hfov / 2.0; i += 0.5)
        {
            pixel_density_deg2.y.push_back(getPixelDensityDeg2(i));
            pixel_density_deg2.x.push_back(i);
        }
        return pixel_density_deg2;
    }

private:
    PixelConeModelConfig config_;
    std::unique_ptr<ConeModel> cone_model_ptr_;

    double f_pix_;
    Graph<double> pixel_densities_deg2_;
    Graph<double> pixels_linear_density_integral_;
};