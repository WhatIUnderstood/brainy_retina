#pragma once
#include "../Cuda/retinastructs.h"
#include "../Utils/interp_utils.h"

#include <opencv2/core.hpp>
#include <stdlib.h>
#include <math.h>
#include <functional>

#include "Utils/convertion.h"

#include "GenericModel.h"
#include "GCellsModel.h"

class MGCellsSim : public GenericModel
{
public:
    struct MGCellsDensityParams
    {
        GCellsModel::GCellsDensityParams gc_params;
        double rm;
    };

    class MGDensityFunction : public DensityFunction
    {
    public:
        MGDensityFunction(const MGCellsDensityParams &params) : params(params), gc_density_f_(params.gc_params)
        {
        }
        double at(double ecc_deg) const override
        {
            return gc_density_f_.at(ecc_deg) / (1 + ecc_deg / params.rm);
        }

    private:
        MGCellsDensityParams params;
        GCellsModel::GCellsDensityFunction gc_density_f_;
    };

    MGCellsSim(const MGCellsDensityParams &params) : GenericModel(std::make_unique<MGDensityFunction>(params), 0, 80, 0.05)
    {
    }
};