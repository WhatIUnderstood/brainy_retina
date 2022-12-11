#pragma once

#include <functional>
#include <memory>

#include "DensityFunction.h"
#include "EccentricityModelInterface.h"
#include "utils/interp_utils.h"

/**
 * @brief Base model that link eccentricity with densities. The retina has a lot of phenomenons that are only depending
 * on the eccentricity
 *
 */
class GenericEccentricityModel : public EccentricityModelInterface {
 public:
  GenericEccentricityModel(std::unique_ptr<DensityFunction> density_function, double min_ecc, double max_ecc,
                           double step)
      : density_function_ptr_(std::move(density_function)) {
    density_graph_ =
        interp_utils::buildGraph([this](double ecc) { return density_function_ptr_->at(ecc); }, min_ecc, max_ecc, step);
    linear_density_integral_graph_ = interp_utils::computeLinearDensityIntegral<double>(density_graph_);
  }

  virtual double getMaxIndex() const override { return static_cast<int>(linear_density_integral_graph_.y.back()); }
  virtual double getMaxEccentricity() const override {
    return static_cast<int>(linear_density_integral_graph_.x.back());
  }

  /**
   * @brief This method should be implemented
   *
   * @param ecc_deg
   * @return double
   */
  virtual double getDensityAt(double ecc_deg) const { return density_function_ptr_->at(ecc_deg); }

  virtual double getIndexAt(double ecc_deg) const override {
    return interp_utils::lin_interp(ecc_deg, linear_density_integral_graph_.x, linear_density_integral_graph_.y, -1,
                                    -1);
  }

  virtual double getEccentricityAt(double element_index) const override {
    return interp_utils::lin_interp(static_cast<double>(element_index), linear_density_integral_graph_.y,
                                    linear_density_integral_graph_.x);
  }

 protected:
  std::unique_ptr<DensityFunction> density_function_ptr_;
  Graph<double> density_graph_;
  Graph<double> linear_density_integral_graph_;
};