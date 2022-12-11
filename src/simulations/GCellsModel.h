#pragma once
#include <math.h>
#include <stdlib.h>

#include <functional>
#include <opencv2/core.hpp>

#include "GenericEccentricityModel.h"
#include "gpu/retinastructs.h"
#include "utils/convertion.h"
#include "utils/interp_utils.h"

/**
 * @brief Ganglionar cells model
 *
 */
class GCellsModel : public GenericEccentricityModel {
 public:
  struct GCellsDensityParams {
    double max_cone_density = 0; // deg2
    double rm;
    double a;
    double r2;
    double re;
  };

  class GCellsDensityFunction : public DensityFunction {
   public:
    GCellsDensityFunction(const GCellsDensityParams &params) : params(params) {}
    double at(double ecc_deg) const override {
      return 2 * params.max_cone_density *
             (params.a * std::pow(1 + ecc_deg / params.r2, -2) + (1 - params.a) * exp(-ecc_deg / params.re));
    }
    GCellsDensityParams params;
  };

  GCellsModel(const GCellsDensityParams &params)
      : GenericEccentricityModel(std::make_unique<GCellsDensityFunction>(params), 0, 80, 0.05) {}
};