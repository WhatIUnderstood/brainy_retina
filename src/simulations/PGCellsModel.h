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
#include "MGCellsSim.h"

class PGCellsModel : public GenericModel
{
public:
    class PCellsDensityFunction : public DensityFunction
    {
    public:
        PCellsDensityFunction(const MGCellsSim::MGCellsDensityParams &params) : gcells_function_(params.gc_params), mgcells_function_(params)
        {
        }
        double at(double ecc_deg) const override
        {
            return 0.04 * gcells_function_.at(ecc_deg) + 0.2 * (gcells_function_.at(ecc_deg) - mgcells_function_.at(ecc_deg));
        }

    private:
        GCellsModel::GCellsDensityFunction gcells_function_;
        MGCellsSim::MGDensityFunction mgcells_function_;
    };

    PGCellsModel(const MGCellsSim::MGCellsDensityParams &params) : GenericModel(std::make_unique<PCellsDensityFunction>(params), 0, 80, 0.05)
    {
    }
};