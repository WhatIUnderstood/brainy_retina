#pragma once
#include "../Cuda/retinastructs.h"
#include "../Utils/interp_utils.h"
#include "../data/data.h"

#include <opencv2/core.hpp>
#include <stdlib.h>
#include <math.h>

#include "Utils/convertion.h"

struct ConeModelConfig
{
    double ph_S_cone_ratio = 5.0 / 100.0;
    double ph_M_cone_ratio = 70.0 / 100.0;
    double ph_L_cone_ratio = 25.0 / 100.0;
};

class ConeModel
{
public:
    ConeModel(const ConeModelConfig &config)
    {
        config_ = config;

        // Use Curcio densities.
        cone_densities_deg2_ = extractConeDensityByDeg2(data::CurcioConeDensities);
        cone_linear_density_integral_ = interp_utils::computeLinearDensityIntegral<double>(cone_densities_deg2_);
    }

    int getConeRadius()
    {
        return static_cast<int>(cone_linear_density_integral_.y.back());
    }

    double getConeAngularPose(double cone_index)
    {
        return interp_utils::lin_interp(static_cast<double>(cone_index), cone_linear_density_integral_.y, cone_linear_density_integral_.x);
    }

    double getConeIndex(double ecc_deg)
    {

        double cone_index = interp_utils::lin_interp(ecc_deg, cone_linear_density_integral_.x, cone_linear_density_integral_.y, -1, -1);

        // if there is no cones the integral stay flat. To prevent to give a cone index in this area
        // TODO handle this better
        // if (std::abs(std::abs(getConeAngularPose(cone_index)) - std::abs(ecc_deg)) > 0.0001)
        // {
        //     return -1.0;
        // }

        return cone_index;
    }

    Graph<double> extractConeDensityByDeg2(const std::vector<std::vector<double>> &source_in_mm2)
    {
        Graph<double> cone_densities_deg2;
        const auto &cone_ecc_deg = source_in_mm2[data::X_DEG];
        for (unsigned int i = 0; i < cone_ecc_deg.size(); i++)
        {
            double cone_density_mm2 = interp_utils::lin_interp(cone_ecc_deg[i], source_in_mm2[data::X_DEG], source_in_mm2[data::NASAL_INDEX]);
            double cone_density_deg2 = convertion::convert_mm2_to_deg2(cone_ecc_deg[i], cone_density_mm2);
            cone_densities_deg2.y.push_back(cone_density_deg2);
            cone_densities_deg2.x.push_back(cone_ecc_deg[i]);
        }
        return cone_densities_deg2;
    }

    Graph<double> computeLinearDensityIntegral(const Graph<double> densities_deg2)
    {
        std::vector<double> cone_linear_density;

        for (const auto &density_deg2 : densities_deg2.y)
        {
            cone_linear_density.push_back(std::sqrt(density_deg2));
            //std::cout << "computeLinearConeDensityIntegral: " << density_deg2 << " " << std::sqrt(density_deg2) << std::endl;
        }

        Graph<double> result;
        result.y = interp_utils::lin_interp_integral(densities_deg2.x, cone_linear_density);
        result.x = densities_deg2.x;
        return result;

        // for (unsigned int i = 0; i < ret.size(); i++)
        // {
        //     std::cout << "computeLinearConeDensityIntegral: i: " << i << " x: " << data::CurcioConeDensities[data::X_DEG][i] << " y: " << ret[i] << std::endl;
        // }
    }

private:
    ConeModelConfig config_;

    Graph<double> cone_densities_deg2_;
    Graph<double> cone_linear_density_integral_;
};