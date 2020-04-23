#pragma once
#include "simulations/PixelConeModel.h"
#include "simulations/ConeModel.h"
#include <iostream>
#include <map>
#include <memory>

#include "Utils/Random.h"
#include "Utils/polar_utils.h"

class ConeLayer
{
public:
    ConeLayer(std::unique_ptr<ConeModel> &cone_model_ptr, std::unique_ptr<PixelConeModel> &pixel_model_ptr, int seed = 1) : cone_model_ptr_(std::move(cone_model_ptr)), pixel_model_ptr_(std::move(pixel_model_ptr)), random(seed)
    {
        int input_width = pixel_model_ptr_->getWidth();
        int input_height = pixel_model_ptr_->getHeight();
        std::cout << "Input dimensions                : " << input_width << " x " << input_height << " pixels" << std::endl;
        std::cout << "Cone model radius               : " << cone_model_ptr_->getMaxIndex() << " cones" << std::endl;
        std::cout << "Cone model max eccentricity     : " << cone_model_ptr_->getEccentricityAt(cone_model_ptr_->getMaxIndex()) << " °" << std::endl;

        double maximum_simulated_cones_radius = cone_model_ptr_->getIndexAt(pixel_model_ptr_->getMaxEccentricity());
        std::cout << "Simulated cone max radius       : " << maximum_simulated_cones_radius << " cones" << std::endl;
        std::cout << "Simulated cone max eccentricity : " << pixel_model_ptr_->getMaxEccentricity() << " °" << std::endl;

        if (maximum_simulated_cones_radius <= 0)
        {
            std::cerr << "initCone: invalid cone source" << std::endl;
            throw std::invalid_argument("invalid cone source");
        }

        //Findout cones dimensions (width and height)
        int cone_width = maximum_simulated_cones_radius * 2;
        cone_width -= cone_width % BLOCK_SIZE; // GPU support BLOCK_SIZE multiples
        int cone_height = cone_model_ptr_->getIndexAt(pixel_model_ptr_->getMaxVEccentricity()) * 2;
        cone_height -= cone_height % BLOCK_SIZE; // GPU support BLOCK_SIZE multiples

        std::cout << "Cone layer dimensions are " << cone_width << " x " << cone_height << std::endl;

        if (cone_height <= 0 || cone_width <= 0)
        {
            std::cerr << "Parameter implies empty cone array" << std::endl;
            return;
        }

        cones_cpu_.cones.resize(cone_width * cone_height);
        cones_cpu_.width = cone_width;
        cones_cpu_.height = cone_height;

        Cone cone;
        double r;

        //cv::Mat mat(cone_width, cone_width, CV_8UC3, cv::Vec3b(255, 255, 255));
        // Map containing info on the ganglionar cells properties
        std::map<std::string, std::vector<float>> description_map_;

        //Default model
        for (int j = 0; j < cone_height; j++)
        {
            for (int i = 0; i < cone_width; i++)
            {
                r = sqrt((cone_width / 2.0 - i) * (cone_width / 2.0 - i) + (cone_height / 2.0 - j) * (cone_height / 2.0 - j));

                cv::Vec2f direction = polar_utils::getDirectionFromCenter(cv::Point(i, j), cv::Size(cone_width, cone_height));
                double cone_angular_pose = cone_model_ptr_->getEccentricityAt(r);
                double pixel_index = pixel_model_ptr_->getIndexAt(cone_angular_pose);
                cv::Point src_pos = polar_utils::getPosition(pixel_index, cv::Size(input_width, input_height), direction);

                if (pixel_index < 0 || src_pos.x > input_width || src_pos.y > input_height ||
                    src_pos.x < 0 || src_pos.y < 0)
                {
                    cone.type = PHOTO_TYPE::NONE;
                }
                else
                {

                    cone.center_x = src_pos.x;
                    cone.center_y = src_pos.y;
                    cone.type = random.weightedRandom<PHOTO_TYPE>({{cone_model_ptr_->config().ph_S_cone_ratio, PHOTO_TYPE::S_CONE}, {cone_model_ptr_->config().ph_M_cone_ratio, PHOTO_TYPE::M_CONE}, {cone_model_ptr_->config().ph_L_cone_ratio, PHOTO_TYPE::L_CONE}});

                    if (j == cone_height / 8)
                    {
                        description_map_["pixel_index"].push_back(pixel_index);
                        description_map_["pixel_index_x"].push_back(i);
                    }
                    else if (j == cone_height / 2)
                    {
                        double angular_pose = cone_model_ptr_->getEccentricityAt(r);
                        description_map_["cone_index_at_half"].push_back(cone_model_ptr_->getIndexAt(angular_pose));
                        description_map_["cone_index_at_half_x"].push_back(i);
                        description_map_["cone_index_at_half_deg"].push_back(angular_pose);
                    }
                }
                cones_cpu_.cones[i + j * cone_width] = cone;
            }
        }
    }

    const ConeModel &coneModel() const
    {
        return *cone_model_ptr_;
    }

    const PixelConeModel &pixelModel() const
    {
        return *pixel_model_ptr_;
    }

    void plotGraphs();

    const Cones &cones() const
    {
        return cones_cpu_;
    }

private:
    // Map containing info on the ganglionar cells properties
    std::map<std::string, std::vector<float>> description_map_;
    Cones cones_cpu_;
    std::unique_ptr<ConeModel> cone_model_ptr_;
    std::unique_ptr<PixelConeModel> pixel_model_ptr_;
    Random random;
};