#pragma once
#include "../Cuda/retinastructs.h"
#include "../Utils/interp_utils.h"

#include <opencv2/core.hpp>
#include <stdlib.h>
#include <math.h>
#include <functional>

#include "Utils/convertion.h"

class MGCellsSim
{
public:
    struct MGCellsConfig
    {
        double max_cone_density = 0; // deg2
        double rm;
        double a;
        double r2;
        double re;
    };

    MGCellsSim(const MGCellsConfig &config)
    {
        config_ = config;
        max_model_ecc_ = 80; // The model fonction has been made to go up to 80°

        std::function<double(double)> midget_density_function = [this](double ecc_deg) { return midgetCellsDensityByDeg2(ecc_deg); };
        auto midget_density_by_deg2 = interp_utils::buildGraph(midget_density_function, 0.0, max_model_ecc_, 0.5);
        midget_linear_density_integral_ = interp_utils::computeLinearDensityIntegral<double>(midget_density_by_deg2);
    }

    double getMaxEccentricity()
    {
        return max_model_ecc_;
    }

    double midgetCellsDensityByDeg2(double ecc_deg)
    {
        return 2 * config_.max_cone_density / (1 + ecc_deg / config_.rm) * (config_.a * std::pow(1 + ecc_deg / config_.r2, -2) + (1 - config_.a) * exp(-ecc_deg / config_.re));
    }

    int
    getTotalRadius()
    {
        return static_cast<int>(midget_linear_density_integral_.y.back());
    }

    double getMidgetCellAngularPose(double midget_index)
    {
        return interp_utils::lin_interp(midget_index, midget_linear_density_integral_.y, midget_linear_density_integral_.x, -1, -1);
    }

private:
    MGCellsConfig config_;
    Graph<double> midget_linear_density_integral_;
    double max_model_ecc_;
};