#pragma once

#include <cmath>

namespace ramp_utils
{
struct RampParameters
{
    double initial_value;
    double final_value;
    double transitional_start;
    double transitional_end;
};

/**
     * @brief Area underRamp function. x represent the area.
     *          ______ <-- final value
     *         /
     *        /        <-- transition
     *   -----         <-- initial value
     *
     *   _____________ <-- 0
     *
     * @param x
     * @param parameters
     * @return double
     */
inline double linearRamp(double x, const RampParameters &parameters)
{
    // If we are in below transitional_start, return initial_value * x
    if (x < parameters.transitional_start)
    {
        return parameters.initial_value;
    }
    else if (x < parameters.transitional_end)
    {
        // We are on the linear increase part of the ramp.
        double steep = (parameters.final_value - parameters.initial_value) / (parameters.transitional_end - parameters.transitional_start);
        return steep * (x - parameters.transitional_start) + parameters.initial_value;
    }
    else
    {
        return parameters.final_value;
    }
}

inline double affineIntegral(double a, double b, double x)
{
    // Integral of ax + b is ax²/2 + bx
    return a * std::pow(x, 2) / 2.0 + b * x;
}

/**
     * @brief Area underRamp function. x represent the area.
     *          ______ <-- final value
     *         /xxxxxx
     *        /xxxxxxx <-- transition
     *   -----xxxxxxxx <-- initial value
     *   xxxxxxxxxxxxx
     *   _____________ <-- 0
     *
     * @param distance_from_center
     * @return double
     */
inline double linearRampIntegral(double x, const RampParameters &parameters)
{
    // If we are in below transitional_start, return initial_value * x
    if (x < parameters.transitional_start)
    {
        // Apply a step of gc_fovea_inter_cells_x
        return x * parameters.initial_value;
    }
    else
    {
        // We are on the linear increase part of the ramp.
        double steep = (parameters.final_value - parameters.initial_value) / (parameters.transitional_end - parameters.transitional_start);
        double initial_aire = parameters.initial_value * parameters.transitional_start;

        if (x < parameters.transitional_end)
        {
            double x_linear = (x - parameters.transitional_start);
            // Integral of ax + b is ax²/2 + bx
            return initial_aire + affineIntegral(steep, parameters.initial_value, x_linear);
        }
        else
        {
            double transitional_dx = (parameters.transitional_end - parameters.transitional_start);
            double final_dx = (x - parameters.transitional_end);

            return initial_aire + affineIntegral(steep, parameters.initial_value, transitional_dx) + parameters.final_value * final_dx;
        }
    }
}
} // namespace ramp_utils