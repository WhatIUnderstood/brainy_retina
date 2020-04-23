#pragma once
#include "../Cuda/retinastructs.h"
#include "../Utils/interp_utils.h"
#include "../data/data.h"

#include <opencv2/core.hpp>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "Utils/convertion.h"
#include "GenericModel.h"

struct ConeModelConfig
{
    double ph_S_cone_ratio = 5.0 / 100.0;
    double ph_M_cone_ratio = 70.0 / 100.0;
    double ph_L_cone_ratio = 25.0 / 100.0;
};

class ConeDensityFunction : public DensityFunction
{
public:
    double at(double ecc_deg) const override
    {
        return convertion::convert_mm2_to_deg2(ecc_deg, std::exp(0.18203247 * std::pow(std::log(ecc_deg + 1.0), 2) + -1.74195991 * std::log(ecc_deg + 1.0) + 12.18370016));
    }
};

class ConeModel : public GenericModel
{
public:
    ConeModel(const ConeModelConfig &config) : GenericModel(std::make_unique<ConeDensityFunction>(), 0, 40, 0.05), config_(config)
    {
        std::cout
            << "Cone density at 0°: " << density_graph_.y[0] << " cones/deg^2" << std::endl;
    }

    const ConeModelConfig &config() const
    {
        return config_;
    }

private:
    ConeModelConfig config_;
    std::function<double(double)> cone_density_function_;

    Graph<double> cone_densities_deg2_;
    Graph<double> cone_linear_density_integral_;
};