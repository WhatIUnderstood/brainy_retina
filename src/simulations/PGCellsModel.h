#pragma once
#include <math.h>
#include <stdlib.h>

#include <functional>
#include <opencv2/core.hpp>

#include "GCellsModel.h"
#include "GenericEccentricityModel.h"
#include "MGCellsSim.h"
#include "gpu/retinastructs.h"
#include "utils/convertion.h"
#include "utils/interp_utils.h"

/**
 * @brief Parasol Ganglionar cell layer
 *
 */
class PGCellsModel : public GenericEccentricityModel {
 public:
  class PCellsDensityFunction : public DensityFunction {
   public:
    PCellsDensityFunction(const MGCellsSim::MGCellsDensityParams &params)
        : gcells_function_(params.gc_params), mgcells_function_(params) {}
    double at(double ecc_deg) const override {
      return 0.04 * gcells_function_.at(ecc_deg) + 0.2 * (gcells_function_.at(ecc_deg) - mgcells_function_.at(ecc_deg));
    }

   private:
    GCellsModel::GCellsDensityFunction gcells_function_;
    MGCellsSim::MGDensityFunction mgcells_function_;
  };

  PGCellsModel(const MGCellsSim::MGCellsDensityParams &params)
      : GenericEccentricityModel(std::make_unique<PCellsDensityFunction>(params), 0, 80, 0.05) {}
};