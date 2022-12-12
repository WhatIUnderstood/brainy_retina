#pragma once
#include <math.h>
#include <stdlib.h>

#include <functional>
#include <opencv2/core.hpp>

#include "GCellsModel.h"
#include "GenericEccentricityModel.h"
#include "gpu/retinastructs.h"
#include "utils/convertion.h"
#include "utils/interp_utils.h"

/**
 * @brief Midget Ganglional cells layer
 *
 */
class MGCellsSim : public GenericEccentricityModel {
 public:
  struct MGCellsDensityParams {
    GCellsModel::GCellsDensityParams gc_params;
    double rm;
  };

  class MGDensityFunction : public DensityFunction {
   public:
    MGDensityFunction(const MGCellsDensityParams &params) : params(params), gc_density_f_(params.gc_params) {}
    double at(double ecc_deg) const override { return gc_density_f_.at(ecc_deg) / (1 + ecc_deg / params.rm); }

   private:
    MGCellsDensityParams params;
    GCellsModel::GCellsDensityFunction gc_density_f_;
  };

  MGCellsSim(const MGCellsDensityParams &params)
      : GenericEccentricityModel(std::make_unique<MGDensityFunction>(params), 0, 80, 0.05) {}
};