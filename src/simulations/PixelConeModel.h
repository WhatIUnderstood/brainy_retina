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

#include "GenericModel.h"

struct PixelConeModelConfig
{
    double camera_hfov = 0; //degree
    int camera_width = 0;
    int camera_height = 0;
};

class PixelConeModel : public GenericModel
{
public:
    PixelConeModel(const PixelConeModelConfig &config) : GenericModel([this, config](double ecc) { return getPixelDensityDeg2(ecc, config); }, 0, config.camera_hfov / 2.0, 0.5)
    {
        config_ = config;
        //cone_model_ptr_ = std::make_unique<ConeModel>(cone_model);

        //f_pix_ = config_.camera_width / (2 * std::tan(config_.camera_hfov * Deg_To_Rad / 2.0));

        // Use pinhole model like
        //pixel_densities_deg2_ = extractPixelDensityByDeg2();
        //pixels_linear_density_integral_ = interp_utils::computeLinearDensityIntegral<double>(pixel_densities_deg2_);
    }

    int getHeight() const
    {
        return config_.camera_height;
    }

    int getWidth() const
    {
        return config_.camera_width;
    }

    // int getMaxRadius()
    // {
    //     return static_cast<int>(config_.camera_hfov / 2.0);
    // }

    double getMaxVEccentricity() const
    {
        return (config_.camera_hfov / (double)config_.camera_width) * (double)config_.camera_height / 2.0;
    }

    double getMaxEccentricity() const override
    {
        return config_.camera_hfov / 2.0;
    }

    // double getMaxVEccentricity()
    // {
    //     return config_.camera_hfov / (double)config_.camera_width * (double)config_.camera_height;
    // }

    // int getSimulatedConeRadius()
    // {
    //     return static_cast<int>(cone_model_ptr_->getIndexAt(config_.camera_hfov / 2.0));
    // }

    // double getConeIndex(double ecc_deg)
    // {
    //     return cone_model_ptr_->getIndexAt(ecc_deg);
    // }

    /**
     * @brief Get the Pixel Index object.
     *
     * @param cone_index
     * @return double
    //  */
    // double getConePixelIndex(int cone_index)
    // {
    //     double cone_angular_pose = cone_model_ptr_->getEccentricityAt(cone_index);
    //     return interp_utils::lin_interp(cone_angular_pose, linear_density_integral_graph_.x, linear_density_integral_graph_.y, -1.0, -1.0);
    // }

    // Graph<double> extractPixelDensityByDeg2()
    // {
    //     Graph<double> pixel_density_deg2;
    //     for (double i = 0; i < config_.camera_hfov / 2.0; i += 0.5)
    //     {
    //         pixel_density_deg2.y.push_back(getPixelDensityDeg2(i));
    //         pixel_density_deg2.x.push_back(i);
    //     }
    //     return pixel_density_deg2;
    // }

    double
    getPixelDensityDeg2(double ecc_deg, const PixelConeModelConfig &config) const
    {
        double f_pix = config.camera_width / (2 * std::tan(config.camera_hfov * Deg_To_Rad / 2.0));
        if (std::abs(ecc_deg) > config.camera_hfov / 2.0)
        {
            return 0;
        }
        return std::pow((f_pix * (std::tan((ecc_deg + 0.5) * Deg_To_Rad) - std::tan((ecc_deg - 0.5) * Deg_To_Rad))), 2);
    }

private:
    PixelConeModelConfig config_;
    //std::unique_ptr<ConeModel> cone_model_ptr_;

    //double f_pix_;
    Graph<double> pixel_densities_deg2_;
    Graph<double> pixels_linear_density_integral_;
};