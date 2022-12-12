#pragma once

#include <math.h>
#include <stdlib.h>

#include <functional>
#include <iostream>
#include <opencv2/core.hpp>

#include "GenericEccentricityModel.h"
#include "data/data.h"
#include "gpu/retinastructs.h"
#include "utils/convertion.h"
#include "utils/interp_utils.h"
#include "ConeModelConfig.h"

/**
 * @brief Cone density function
 *
 */
class ConeDensityFunction : public DensityFunction {
 public:
  /**
   * @brief Return the cone density at the given eccentricity
   *
   * @param ecc_deg eccentricity in degree
   * @return double density
   */
  double at(double ecc_deg) const override {
    return convertion::convert_mm2_to_deg2(ecc_deg, std::exp(0.18203247 * std::pow(std::log(ecc_deg + 1.0), 2) +
                                                             -1.74195991 * std::log(ecc_deg + 1.0) + 12.18370016));
  }
};

/**
 * @brief Cone Model class
 *
 */
class ConeModel : public GenericEccentricityModel {
 public:
  ConeModel(const ConeModelConfig &config)
      : GenericEccentricityModel(std::make_unique<ConeDensityFunction>(), 0, 40, 0.05), config_(config) {
    std::cout << "Cone density at 0°: " << density_graph_.y[0] << " cones/deg^2" << std::endl;
  }

  const ConeModelConfig &config() const { return config_; }

 private:
  ConeModelConfig config_;
  std::function<double(double)> cone_density_function_;

  Graph<double> cone_densities_deg2_;
  Graph<double> cone_linear_density_integral_;
};