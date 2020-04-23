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

class PixelDensityFunction : public DensityFunction
{
public:
    PixelDensityFunction(const PixelConeModelConfig &config) : config_(config)
    {
        f_pix = config.camera_width / (2 * std::tan(config.camera_hfov * Deg_To_Rad / 2.0));
    }

    double at(double ecc_deg) const override
    {
        if (std::abs(ecc_deg) > config_.camera_hfov / 2.0)
        {
            return 0;
        }
        return std::pow((f_pix * (std::tan((ecc_deg + 0.5) * Deg_To_Rad) - std::tan((ecc_deg - 0.5) * Deg_To_Rad))), 2);
    }
    double f_pix;
    PixelConeModelConfig config_;
};

class PixelConeModel : public GenericModel
{
public:
    PixelConeModel(const PixelConeModelConfig &config) : GenericModel(std::make_unique<PixelDensityFunction>(config), 0, config.camera_hfov / 2.0, 0.5), config_(config)
    {
    }

    int getHeight() const
    {
        return config_.camera_height;
    }

    int getWidth() const
    {
        return config_.camera_width;
    }

    double getMaxVEccentricity() const
    {
        return (config_.camera_hfov / (double)config_.camera_width) * (double)config_.camera_height / 2.0;
    }

    double getMaxEccentricity() const override
    {
        return config_.camera_hfov / 2.0;
    }

    // double
    // getPixelDensityDeg2(double ecc_deg, const PixelConeModelConfig &config) const
    // {
    //     double f_pix = config.camera_width / (2 * std::tan(config.camera_hfov * Deg_To_Rad / 2.0));
    //     if (std::abs(ecc_deg) > config.camera_hfov / 2.0)
    //     {
    //         return 0;
    //     }
    //     return std::pow((f_pix * (std::tan((ecc_deg + 0.5) * Deg_To_Rad) - std::tan((ecc_deg - 0.5) * Deg_To_Rad))), 2);
    // }

private:
    PixelConeModelConfig config_;
    //std::unique_ptr<ConeModel> cone_model_ptr_;

    //double f_pix_;
    Graph<double> pixel_densities_deg2_;
    Graph<double> pixels_linear_density_integral_;
};