#pragma once
#include <cmath>

constexpr double Deg_To_Rad = M_PI / 180.0;

namespace convertion
{
inline double convert_mm2_to_deg2(double ecc_deg, double density_mm2)
{
    double a = 0.0752 + 5.846 * std::pow(10, -5) * std::pow(ecc_deg, 1) - 1.064 * std::pow(10, -5) * std::pow(ecc_deg, 2) + 4.116 * std::pow(10, -8) * std::pow(ecc_deg, 3);
    return a * density_mm2;
}

} // namespace convertion