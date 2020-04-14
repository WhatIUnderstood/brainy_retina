#pragma once
#include <cmath>

constexpr double Deg_to_rad = M_PI / 180.0;

namespace convertion
{
inline double convert_mm2_to_deg2(double ecc_deg, double mm2)
{
    double a = 0.0752 + 5.846 * std::pow(10, -5) * std::pow(ecc_deg, 1) - 1.064 * std::pow(10, -5) * std::pow(ecc_deg, 2) + 4.116 * std::pow(10, -8) * std::pow(ecc_deg, 3);
    return a * mm2;
}
} // namespace convertion