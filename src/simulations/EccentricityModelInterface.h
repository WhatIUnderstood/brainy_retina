#pragma once

/**
 * @brief Interface that link
 *      - eccentricities and densities
 *      - eccentricities with a linear representation
 * The retina has a lot of phenomenons that are only depending on the eccentricity
 */
class EccentricityModelInterface {
 public:
  /**
   * @brief Get the Max Eccentricity of the current model
   *
   * @return double max eccentricity in degrees
   */
  virtual double getMaxEccentricity() const = 0;

  /**
   * @brief Get the Density at the given eccentricity
   *
   * @param ecc_deg eccentricity in degrees
   * @return double density
   */
  virtual double getDensityAt(double ecc_deg) const = 0;

  /**
   * @brief Get the layer Index corresponding to the given eccentricity
   *
   * @param ecc_deg eccentricity in degrees
   * @return double layer index
   */
  virtual double getIndexAt(double ecc_deg) const = 0;

  /**
   * @brief Get the Max Index of the layer
   *
   * @return double layer index
   */
  virtual double getMaxIndex() const = 0;

  /**
   * @brief Get the Eccentricity At the given layer index
   *
   * @param index layer index
   * @return double eccentricity in degrees
   */
  virtual double getEccentricityAt(double index) const = 0;
};