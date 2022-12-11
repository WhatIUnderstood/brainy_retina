#pragma once

/**
 * @brief Density function interface
 *
 */
class DensityFunction {
 public:
  /**
   * @brief Return a density at the given eccentricity
   *
   * @param ecc_deg eccentricity in degree
   * @return double density
   */
  virtual double at(double ecc) const = 0;
};
