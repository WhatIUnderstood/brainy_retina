#pragma once
#include "ModelInterface.h"
#include <functional>

#include "Utils/interp_utils.h"

class GenericModel : public ModelInterface
{
public:
    GenericModel(std::function<double(double)> density_function, double min_ecc, double max_ecc, double step) : density_function_(density_function)
    {
        density_graph_ = interp_utils::buildGraph(density_function_, min_ecc, max_ecc, step);
        linear_density_integral_graph_ = interp_utils::computeLinearDensityIntegral<double>(density_graph_);
    }

    virtual double getMaxIndex() const override
    {
        return static_cast<int>(linear_density_integral_graph_.y.back());
    }
    virtual double getMaxEccentricity() const override
    {
        return static_cast<int>(linear_density_integral_graph_.x.back());
    }

    virtual double getDensityAt(double ecc_deg) const override
    {
        return density_function_(ecc_deg);
    }

    virtual double getIndexAt(double ecc_deg) const override
    {
        return interp_utils::lin_interp(ecc_deg, linear_density_integral_graph_.x, linear_density_integral_graph_.y, -1, -1);
    }

    virtual double getEccentricityAt(double element_index) const override
    {
        return interp_utils::lin_interp(static_cast<double>(element_index), linear_density_integral_graph_.y, linear_density_integral_graph_.x);
    }

protected:
    std::function<double(double)> density_function_;

    Graph<double> density_graph_;
    Graph<double> linear_density_integral_graph_;
};