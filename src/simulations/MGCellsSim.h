#pragma once
#include "../Cuda/retinastructs.h"
#include "../Utils/interp_utils.h"

#include <opencv2/core.hpp>
#include <stdlib.h>
#include <math.h>
#include <functional>

#include "Utils/convertion.h"

#include "GenericModel.h"

class MGCellsSim : public GenericModel
{
public:
    struct MGCellsDensityParams
    {
        double max_cone_density = 0; // deg2
        double rm;
        double a;
        double r2;
        double re;
    };

    struct MGCellsConfig
    {
        MGCellsDensityParams density_params;
    };

    MGCellsSim(const MGCellsConfig &config) : GenericModel([this, &config](double ecc) { return this->getmGCDensityAt(ecc, config.density_params); }, 0, 80, 0.05)
    {
        config_ = config;
    }

private:
    double getmGCDensityAt(double ecc_deg, const MGCellsDensityParams &params)
    {
        return 2 * params.max_cone_density / (1 + ecc_deg / params.rm) * (params.a * std::pow(1 + ecc_deg / params.r2, -2) + (1 - params.a) * exp(-ecc_deg / params.re));
    }

private:
    MGCellsConfig config_;
};