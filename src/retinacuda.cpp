#include "retinacuda.h"
#include <opencv2/core/cuda_types.hpp>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda.hpp>
#include "Cuda/cuda_image.h"
#include "cuda_runtime.h"
#include "Cuda/cuda_retina_kernels.cuh"

#include "data/data.h"
#include "Utils/interp_utils.h"

#include <iostream>

#include "simulations/ConeModel.h"
#include "simulations/PixelConeModel.h"

#ifdef WITH_MATPLOTLIB
#include "matplotlib_cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

//Human mgc
float excentricity(float r)
{
    float dgf0 = 30000; // density at r0
    float ak = 0.9729;  // first term weight
    float r2k = 1.084;
    float rek = 7.633;

    //http://jov.arvojournals.org/article.aspx?articleid=2279458#87788067
    return dgf0 * (ak * pow((1 + r / r2k), -2) + (1 - ak) * exp(-r / rek));
}

class RetinaCudaException : public std::exception
{
public:
    RetinaCudaException(std::string message)
    {
        this->message = message;
    }

    // exception interface
public:
    const char *what() const throw()
    {
        return message.c_str();
    }

private:
    std::string message;
};

RetinaCuda::RetinaCuda(int gpu)
{
    gpuCells = 0;
    //cuda_stream_ptr.reset(new cudaStream_t());
    //*(cudaStream_t*)cuda_stream_ptr = cudaStreamDefault;

    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    //int count = cv::cuda::getCudaEnabledDeviceCount();

    if (count == 0 || err != cudaSuccess)
    {

        throw RetinaCudaException(std::string("No gpu avaliable: ") + cudaGetErrorString(err));
    }
    else if (count < gpu)
    {
        throw RetinaCudaException("Gpu not avaliable");
    }

    //err = cudaSetDevice(gpu)
    cv::cuda::setDevice(gpu);

    cv::cuda::DeviceInfo deviceInfos(gpu);

    std::cout << "Using gpu: " << deviceInfos.name() << std::endl;
}

RetinaCuda::~RetinaCuda()
{
}

void RetinaCuda::initRetina(Parameters param)
{
    parameters = param;
    midget_gc_ramp_.initial_value = param.gc_fovea_inter_cells_distance;
    midget_gc_ramp_.final_value = param.gc_max_inter_cells_distance;
    midget_gc_ramp_.transitional_start = param.gc_fovea_radius;
    midget_gc_ramp_.transitional_end = param.gc_max_cones_by_cell_radius;

    midget_gc_field_ramp_.initial_value = param.gc_fovea_cones_by_cell;
    midget_gc_field_ramp_.final_value = param.gc_max_cones_by_cell;
    midget_gc_field_ramp_.transitional_start = param.gc_fovea_radius;
    midget_gc_field_ramp_.transitional_end = param.gc_max_cones_by_cell_radius;

    pixel_per_cone_ramp_.initial_value = param.ph_fovea_pixels_by_cone;
    pixel_per_cone_ramp_.final_value = param.ph_max_pixels_by_cone;
    pixel_per_cone_ramp_.transitional_start = param.ph_fovea_radius;
    pixel_per_cone_ramp_.transitional_end = param.ph_max_pixels_by_cone_radius;

    cone_model_ptr_ = std::make_unique<ConeModel>(parameters.ph_config);
    photo_sim_ptr_ = std::make_unique<PixelConeModel>(parameters.pix_config, *cone_model_ptr_);

    //parasol_gc_ramp_;
    initCone(param.input_width, param.input_height);
    initGanglionarCells(cones_cpu_.width, cones_cpu_.height);
    initSelectiveCells();
}

std::vector<Ganglionar> RetinaCuda::initGanglionarCells(int conesWidth, int conesHeight)
{
    std::vector<Ganglionar> cellsCPU;

    MGCellsSim::MGCellsConfig config;
    config.a = 0.9729;
    config.r2 = 1.084;
    config.re = 7.633;
    config.rm = 41.03;
    config.max_cone_density = 14804;
    MGCellsSim midgetGCells(config);

    std::cout << "midget cells model radius: " << midgetGCells.getTotalRadius() << " cells" << std::endl;
    std::cout << "midget cells model max eccentricity: " << midgetGCells.getMaxEccentricity() << " °" << std::endl;

    // for (unsigned int i = 0; i < midgetGCells.getTotalRadius(); i++)
    // {

    //     std::cout << "midget i: " << i << " angle: " << midgetGCells.getMidgetCellAngularPose(i)
    //               << " cone index:" << cone_model_ptr_->getConeIndex(midgetGCells.getMidgetCellAngularPose(i))
    //               << " cone angle:" << cone_model_ptr_->getConeAngularPose(cone_model_ptr_->getConeIndex(midgetGCells.getMidgetCellAngularPose(i)))
    //               << std::endl;
    //     if (i > 1500)
    //     {
    //         break;
    //     }
    // }

    //Findout cellWidth and cellHeight that fit Pixcone layer
    int cellsWidth = -1;
    int cellsHeight = -1;

    //Get GC width
    for (int r = 0; r < midgetGCells.getTotalRadius() && cellsWidth < 0; r++)
    {
        cv::Vec2f direction(1, 0);

        //double mgcell_ecc = midgetGCells.getMidgetCellAngularPose(r);
        auto ganglionar_cell_central_cone_index = photo_sim_ptr_->getConeIndex(midgetGCells.getMidgetCellAngularPose(r));
        //cv::Point src_pos = getPosition(ganglionar_cell_central_cone_index, cv::Size(conesWidth, conesHeight), direction);
        //cv::Point src_pos = getPosition(mgc_position_mapping(r), cv::Size(conesWidth, conesHeight), direction);
        //if (mgcell_ecc > photo_sim_ptr_->getMaxEccentricity())
        if (ganglionar_cell_central_cone_index > conesWidth / 2)
        {
            cellsWidth = (r - 1) * 2;
            cellsWidth = cellsWidth - cellsWidth % BLOCK_SIZE;
        }
        else if (r == midgetGCells.getTotalRadius() - 1)
        {
            cellsWidth = midgetGCells.getTotalRadius() / 2;
            cellsWidth = cellsWidth - cellsWidth % BLOCK_SIZE;
        }
    }

    //Get cone height
    for (int r = 0; r < midgetGCells.getTotalRadius() && cellsHeight < 0; r++)
    {
        auto ganglionar_cell_central_cone_index = photo_sim_ptr_->getConeIndex(midgetGCells.getMidgetCellAngularPose(r));
        //cv::Vec2f direction(0, 1);
        //cv::Point src_pos = getPosition(ganglionar_cell_central_cone_index, cv::Size(conesWidth, conesHeight), direction);
        //cv::Point src_pos = getPosition(mgc_position_mapping(r), cv::Size(conesWidth, conesHeight), direction);
        // if (src_pos.x >= conesWidth || src_pos.y >= conesHeight ||
        //     src_pos.x < 0 || src_pos.y < 0 || ganglionar_cell_central_cone_index < 0)
        // {
        //     cellsHeight = (r - 1) * 2;
        //     cellsHeight = cellsHeight - cellsHeight % BLOCK_SIZE;
        // }
        //double mgcell_ecc = midgetGCells.getMidgetCellAngularPose(r);
        //cv::Point src_pos = getPosition(ganglionar_cell_central_cone_index, cv::Size(conesWidth, conesHeight), direction);
        //cv::Point src_pos = getPosition(mgc_position_mapping(r), cv::Size(conesWidth, conesHeight), direction);
        if (ganglionar_cell_central_cone_index > conesHeight / 2)
        //if (mgcell_ecc > photo_sim_ptr_->getMaxVEccentricity())
        {
            cellsHeight = (r - 1) * 2;
            cellsHeight = cellsHeight - cellsHeight % BLOCK_SIZE;
        }
        else if (r == midgetGCells.getTotalRadius() - 1)
        {
            cellsHeight = midgetGCells.getTotalRadius() / 2;
            cellsHeight = cellsHeight - cellsHeight % BLOCK_SIZE;
        }
    }

    if (cellsHeight <= 0 || cellsWidth <= 0)
    {
        std::cerr << "Parameter implies empty cone array" << std::endl;
        throw std::invalid_argument("Parameter implies empty cone array");
    }

    cellsArrayHeight = cellsHeight; // - BLOCK_SIZE * 38;
    cellsArrayWidth = cellsWidth;

    std::cout << "Pix midget cells radius: " << cellsWidth / 2 << " cells" << std::endl;
    std::cout << "Pix midget cells max eccentricity: " << midgetGCells.getMidgetCellAngularPose(cellsWidth / 2) << " °" << std::endl;

    std::cout << "midget dimensions: " << cellsArrayWidth << " x " << cellsArrayHeight << std::endl; //" midgetGCells.getTotalRadius()

    cellsCPU.resize(cellsArrayWidth * cellsArrayHeight);

    Ganglionar cell;
    double r;
    double r_next;
    double ganglionarExternalRadius;
    double ganglionarInternalRadius;

    //cv::Mat mat(cellsArrayHeight, cellsArrayWidth, CV_8UC3, cv::Vec3b(255, 255, 255));

    // Map containing info on the ganglionar cells properties
    std::map<std::string, std::vector<float>> description_map;

    setRandomSeed(parameters.random_seed);
    //Default model
    for (int j = 0; j < cellsArrayHeight; j++)
    {
        for (int i = 0; i < cellsArrayWidth; i++)
        {
            r = getDistanceFromCenter(i, j, cellsArrayWidth, cellsArrayHeight);

            r_next = r + 1.0;

            const auto midget_angular_pose = midgetGCells.getMidgetCellAngularPose(r);
            const auto next_midget_angular_pose = midgetGCells.getMidgetCellAngularPose(r_next);

            const auto midget_central_cone = photo_sim_ptr_->getConeIndex(midget_angular_pose);
            const auto next_midget_central_cone = photo_sim_ptr_->getConeIndex(next_midget_angular_pose);

            //std::cout << "next_midget_angular_pose: " << next_midget_angular_pose << std::endl;
            if (midget_angular_pose < 0 || next_midget_angular_pose < 0 || next_midget_central_cone < 0 || midget_central_cone < 0 || std::abs(midget_central_cone - next_midget_central_cone) < 0.1)
            {
                cell.type = GC_RESPONSE_TYPE::NONE;
                cellsCPU[i + j * cellsArrayWidth] = cell;
                continue;
            }

            //int linearReduction = 6;

            ganglionarExternalRadius = MAX(0.5, std::fabs(next_midget_central_cone - midget_central_cone) / std::sqrt(2.0)); // ON and OFF are overlapping
            // ganglionarExternalRadius = mgc_dentric_coverage(r);
            // std::cout
            //     << "ganglionarExternalRadius: " << ganglionarExternalRadius << " r:" << r << " midgetGCells.getMidgetCellAngularPose(r)" << midgetGCells.getMidgetCellAngularPose(r) << " midget_central_cone: " << midget_central_cone << " next_midget_central_cone:" << next_midget_central_cone << std::endl;
            ganglionarInternalRadius = MAX(0.5, 0.33 * ganglionarExternalRadius);
            cv::Vec2f direction = r == 0 ? cv::Vec2f(1, 0) : getDirectionFromCenter(cv::Point(i, j), cv::Size(cellsArrayWidth, cellsArrayHeight));
            cv::Point src_pos = getPosition(midget_central_cone, cv::Size(cones_cpu_.width, cones_cpu_.height), direction);

            //check if cone is valid
            int cone_key = src_pos.x + src_pos.y * cones_cpu_.width;
            if (cone_key >= cones_cpu_.cones.size() || src_pos.x < 0 || src_pos.x >= cones_cpu_.width || src_pos.y < 0 || src_pos.y >= cones_cpu_.height || cones_cpu_.cones[cone_key].type == PHOTO_TYPE::NONE)
            {
                cell.type = GC_RESPONSE_TYPE::NONE;
                cellsCPU[i + j * cellsArrayWidth] = cell;
                continue;
            }

            //cv::Point src_pos = getPosition(mgc_position_mapping(r), cv::Size(cones_cpu_.width, cones_cpu_.height), direction);
            cell.center_x = src_pos.x;
            cell.center_y = src_pos.y;
            cell.extern_radius = ganglionarExternalRadius;
            cell.intern_radius = ganglionarInternalRadius;
            cell.type = i % 2 == 1 ? GC_RESPONSE_TYPE::ON : GC_RESPONSE_TYPE::OFF;
            cellsCPU[i + j * cellsArrayWidth] = cell;

            //memcpy( cellsArrayGPU + i+j*cellsArrayWidth,&cell, sizeof(cell));
            //cellsArrayGPU[i+j*cellsArrayWidth] = cell;

            if (j == cellsArrayHeight / 8)
            {
                description_map["gcm_radius_at_eighth"].push_back(next_midget_angular_pose - midget_angular_pose);
                //description_map["gcm_radius_at_eighth"].push_back(next_midget_central_cone - midget_central_cone);
                description_map["gcm_radius_at_eighth_x"].push_back(i);
                // description_map["gcm_radius_next_at_eighth"].push_back(next_midget_central_cone);
                // description_map["gcm_radius_next_at_eighth_x"].push_back(i);

                description_map["gc_external_radius_at_eighth"].push_back(ganglionarExternalRadius);
                description_map["gc_external_radius_at_eighth_x"].push_back(i);
                description_map["gc_external_radius_at_eighth_deg"].push_back(midget_angular_pose);

                description_map["gc_midget_angular_eighth_pose"].push_back(midget_angular_pose);
                description_map["gc_midget_angular_eighth_pose_x"].push_back(i);

                description_map["gc_midget_cone_eighth_pose"].push_back(midget_central_cone);
                description_map["gc_midget_cone_eighth_pose_x"].push_back(i);
                description_map["gc_midget_cone_eighth_pose_deg"].push_back(midget_angular_pose);
            }
            else if (j == cellsArrayHeight / 2)
            {
                description_map["gc_external_radius_at_half"].push_back(ganglionarExternalRadius);
                description_map["gc_external_radius_at_half_x"].push_back(i);
                description_map["gc_external_radius_at_half_deg"].push_back(midget_angular_pose);

                description_map["gc_midget_angular_half_pose"].push_back(midget_angular_pose);
                description_map["gc_midget_angular_half_pose_x"].push_back(i);

                description_map["gc_midget_cone_half_pose"].push_back(midget_central_cone);
                description_map["gc_midget_cone_half_pose_x"].push_back(i);
                description_map["gc_midget_cone_half_pose_deg"].push_back(midget_angular_pose);

                description_map["gcm_density"].push_back(midgetGCells.midgetCellsDensityByDeg2(midget_angular_pose));
                description_map["cone_density"].push_back(cone_model_ptr_->getDensityAt(midget_angular_pose));
                description_map["gcm_cone_ratio"].push_back(midgetGCells.midgetCellsDensityByDeg2(midget_angular_pose) / cone_model_ptr_->getDensityAt(midget_angular_pose));
                description_map["gcm_density_deg"].push_back(midget_angular_pose);
            }
            // //Display stuff
            // if (j == cellsArrayHeight / 2)
            // {
            //     cv::Vec3b red = cv::Vec3b(255, 0, 0);
            //     cv::Vec3b black = cv::Vec3b(0, 0, 0);
            //     mat.at<cv::Vec3b>(cv::Point(i, j + ganglionarInternalRadius)) = black;
            //     mat.at<cv::Vec3b>(cv::Point(i, j + ganglionarExternalRadius)) = cv::Vec3b(0, 255, 0);
            //     mat.at<cv::Vec3b>(cv::Point(i, j)) = red;
            // }
        }
    }

#ifdef WITH_MATPLOTLIB
    plt::figure();
    plt::named_plot("radius 1/8th", description_map["gcm_radius_at_eighth_x"], description_map["gcm_radius_at_eighth"]);
    //plt::named_plot("radius newt 1/8th", description_map["gcm_radius_next_at_eighth_x"], description_map["gcm_radius_next_at_eighth"]);
    plt::title("Test");
    plt::legend();

    plt::figure();
    plt::named_plot("GCm angular pos 1/8th", description_map["gc_midget_angular_eighth_pose_x"], description_map["gc_midget_angular_eighth_pose"]);
    plt::named_plot("GCm angular pos 1/2th", description_map["gc_midget_angular_half_pose_x"], description_map["gc_midget_angular_half_pose"]);
    plt::title("Ganglionar cells angular pose");
    plt::legend();
    //
    plt::figure();
    plt::named_plot("GC ext radius 1/8th", description_map["gc_external_radius_at_eighth_x"], description_map["gc_external_radius_at_eighth"]);
    plt::named_plot("GC ext radius 1/2th", description_map["gc_external_radius_at_half_x"], description_map["gc_external_radius_at_half"]);
    plt::title("Ganglionar cells radius");
    plt::legend();

    plt::figure();
    plt::named_plot("GC ext radius 1/8th", description_map["gc_external_radius_at_eighth_deg"], description_map["gc_external_radius_at_eighth"]);
    plt::named_plot("GC ext radius 1/2th", description_map["gc_external_radius_at_half_deg"], description_map["gc_external_radius_at_half"]);
    plt::title("Ganglionar cells radius (deg)");
    plt::legend();

    plt::figure();
    plt::named_plot("GCm cone center 1/8th", description_map["gc_midget_cone_eighth_pose_x"], description_map["gc_midget_cone_eighth_pose"]);
    plt::named_plot("GCm cone center 1/2th", description_map["gc_midget_cone_half_pose_x"], description_map["gc_midget_cone_half_pose"]);
    plt::title("Ganglionar cells cone center");
    plt::legend();

    plt::figure();
    plt::named_plot("GCm cone center 1/8th", description_map["gc_midget_cone_eighth_pose_deg"], description_map["gc_midget_cone_eighth_pose"]);
    plt::named_plot("GCm cone center 1/2th", description_map["gc_midget_cone_half_pose_deg"], description_map["gc_midget_cone_half_pose"]);
    plt::title("Ganglionar cells cone center (deg)");
    plt::legend();

    plt::figure();
    plt::named_plot("GCm density 1/2th", description_map["gcm_density_deg"], description_map["gcm_density"]);
    plt::named_plot("Cone density at 1/2th", description_map["gcm_density_deg"], description_map["cone_density"]);
    plt::title("GCm vs Cone densities (deg)");
    plt::legend();

    plt::figure();
    plt::named_plot("GCm cone ratio 1/2th", description_map["gcm_density_deg"], description_map["gcm_cone_ratio"]);
    plt::title("GCm / Cone ratio (deg)");
    plt::legend();

    plt::show();
#endif

    //cv::imshow("Cells", mat);
    initCellsGpu(cellsCPU, cellsArrayWidth, cellsArrayHeight);
    //free(cellsArrayGPU);
    return cellsCPU;
}

std::vector<Point> RetinaCuda::initSelectiveCells()
{
    magnoMappingSrc.clear();
    magnoMappingDst.clear();
    // int x = 0;
    // int y = 0;
    // int max_x = 0;

    int width = cones_cpu_.width / 2;
    width -= width % BLOCK_SIZE;
    int height = cones_cpu_.height / 2;
    height -= height % BLOCK_SIZE;
    for (unsigned int h = 0; h < height; h++)
    {
        //x = 0;
        for (unsigned int w = 0; w < width; w++)
        {
            magnoMappingSrc.push_back(Point(w * 2, h * 2));
            magnoMappingDst.push_back(Point(w, h));
            // if (x > max_x)
            // {
            //     max_x = x;
            // }
            //x++;

            //            if(w >= coneMarge.at(h) && w <= conesArrayWidth-coneMarge.at(h)){
            //                if(sqrt((float)((h-conesArrayHeight/2.0)*(h-conesArrayHeight/2.0)+(w-conesArrayWidth/2.0)*(w-conesArrayWidth/2.0))) > parameters.ph_fovea_radius){

            //                    magnoMappingSrc.push_back(Point(w,h));
            //                    magnoMappingDst.push_back(Point(x,h));
            //                    if(x>max_x){
            //                        max_x = x;
            //                    }
            //                    x++;
            //                }
            //            }
        }
        //y++;
    }
    directive_width = width;
    directive_height = height;
    initDirectiveGpu(magnoMappingSrc, magnoMappingDst);

    return magnoMappingSrc;
}

void RetinaCuda::initCone(int inputWidth, int inputHeight)
{
    ConeModel cone_model(parameters.ph_config);
    std::cout << "Cone model radius          : " << cone_model_ptr_->getConeRadius() << " cones" << std::endl;
    std::cout << "Cone model max eccentricity: " << cone_model_ptr_->getConeAngularPose(cone_model_ptr_->getConeRadius()) << " °" << std::endl;
    std::cout << "Pixel Cone model radius    : " << photo_sim_ptr_->getSimulatedConeRadius() << " pixcones" << std::endl;
    std::cout << "Pixel Cone max eccentricity: " << photo_sim_ptr_->getMaxEccentricity() << " °" << std::endl;

    //Findout cones dimensions (width and height)
    int coneWidth = -1;
    int coneHeight = -1;

    if (photo_sim_ptr_->getSimulatedConeRadius() <= 0)
    {
        std::cerr << "initCone: invalid cone source" << std::endl;
        throw std::invalid_argument("invalid cone source");
    }

    //Get acheivable simulated cones width
    for (int r = 0; r < photo_sim_ptr_->getSimulatedConeRadius() && coneWidth < 0; r++)
    {
        cv::Vec2f direction(1, 0);
        double pixel_index = photo_sim_ptr_->getPixelIndex(r);

        cv::Point src_pos = getPosition(pixel_index, cv::Size(inputWidth, inputHeight), direction);

        // If no matching pixel, stop
        if (pixel_index < 0 || src_pos.x >= inputWidth || src_pos.y >= inputHeight ||
            src_pos.x < 0 || src_pos.y < 0)
        {
            coneWidth = (r - 1) * 2;
            coneWidth = coneWidth - coneWidth % BLOCK_SIZE;
        }

        // if (pixel_index >= 0) // a pixel has matched
        // {
        //     cv::Point src_pos = getPosition(pixel_index, cv::Size(inputWidth, inputHeight), direction);
        //     // cv::Point src_pos = getPosition(cone_distance_mapping(r), cv::Size(inputWidth, inputHeight), direction); // TODO generic keep
        //     if (src_pos.x >= inputWidth || src_pos.y >= inputHeight ||
        //         src_pos.x < 0 || src_pos.y < 0)
        //     {
        //         coneWidth = (r - 1) * 2;
        //         coneWidth = coneWidth - coneWidth % BLOCK_SIZE;
        //     }
        // }
        // else
        // {
        // }
    }

    if (coneWidth < 0)
    {
        coneWidth = photo_sim_ptr_->getSimulatedConeRadius();
    }

    //Get cones height
    for (int r = 0; r < photo_sim_ptr_->getSimulatedConeRadius() && coneHeight < 0; r++)
    {
        cv::Vec2f direction(0, 1);
        double pixel_index = photo_sim_ptr_->getPixelIndex(r);
        cv::Point src_pos = getPosition(pixel_index, cv::Size(inputWidth, inputHeight), direction);

        // If no matching pixel, stop
        if (pixel_index < 0 || src_pos.x >= inputWidth || src_pos.y >= inputHeight ||
            src_pos.x < 0 || src_pos.y < 0)
        {
            coneHeight = (r - 1) * 2;
            coneHeight = coneHeight - coneHeight % BLOCK_SIZE;
        }

        // cv::Vec2f direction(0, 1);
        // //cv::Point src_pos = getPosition(cone_distance_mapping(r), cv::Size(inputWidth, inputHeight), direction);
        // cv::Point src_pos = getPosition(photo_sim_ptr_->getPixelIndex(r), cv::Size(inputWidth, inputHeight), direction);
        // if (src_pos.x >= inputWidth || src_pos.y >= inputHeight ||
        //     src_pos.x < 0 || src_pos.y < 0)
        // {
        //     coneHeight = (r - 1) * 2;
        //     coneHeight = coneHeight - coneHeight % BLOCK_SIZE;
        // }
    }

    std::cout << "Cone map dimension are " << coneWidth << " x " << coneHeight << std::endl;

    if (coneHeight <= 0 || coneWidth <= 0)
    {
        std::cerr << "Parameter implies empty cone array" << std::endl;
        return;
    }

    cones_cpu_.cones.resize(coneWidth * coneHeight);
    cones_cpu_.width = coneWidth;
    cones_cpu_.height = coneHeight;

    Cone cone;
    double r;

    //cv::Mat mat(coneWidth, coneWidth, CV_8UC3, cv::Vec3b(255, 255, 255));
    // Map containing info on the ganglionar cells properties
    std::map<std::string, std::vector<float>> description_map;

    //Default model
    setRandomSeed(parameters.random_seed);
    for (int j = 0; j < coneHeight; j++)
    {
        bool beginLine = false;
        for (int i = 0; i < coneWidth; i++)
        {
            //int linearReduction = 6;
            r = sqrt((coneWidth / 2.0 - i) * (coneWidth / 2.0 - i) + (coneHeight / 2.0 - j) * (coneHeight / 2.0 - j));
            //ganglionarExternalRadius = cone_coverage(r);
            //ganglionarInternalRadius = MAX(1.0,ganglionarExternalRadius/2.0);
            cv::Vec2f direction = getDirectionFromCenter(cv::Point(i, j), cv::Size(coneWidth, coneHeight));
            double pixel_index = photo_sim_ptr_->getPixelIndex(r);
            cv::Point src_pos = getPosition(pixel_index, cv::Size(parameters.input_width, parameters.input_height), direction);
            //cv::Point src_pos = getPosition(cone_distance_mapping(r), cv::Size(parameters.input_width, parameters.input_height), direction);

            if (pixel_index < 0 || src_pos.x >= parameters.input_width || src_pos.y >= parameters.input_height ||
                src_pos.x < 0 || src_pos.y < 0)
            {
                cone.type = PHOTO_TYPE::NONE;
            }
            else
            {

                cone.center_x = src_pos.x;
                cone.center_y = src_pos.y;
                cone.type = weightedRandom<PHOTO_TYPE>({{parameters.ph_S_cone_ratio, PHOTO_TYPE::S_CONE}, {parameters.ph_M_cone_ratio, PHOTO_TYPE::M_CONE}, {parameters.ph_L_cone_ratio, PHOTO_TYPE::L_CONE}});

                if (j == coneHeight / 8)
                {
                    description_map["pixel_index"].push_back(pixel_index);
                    description_map["pixel_index_x"].push_back(i);
                }
                else if (j == coneHeight / 2)
                {
                    double angular_pose = cone_model_ptr_->getConeAngularPose(r);
                    description_map["cone_index_at_half"].push_back(cone_model_ptr_->getConeIndex(angular_pose));
                    description_map["cone_index_at_half_x"].push_back(i);
                    description_map["cone_index_at_half_deg"].push_back(angular_pose);
                }
                // //Display stuff
                // if(j == coneHeight/2){
                //     cv::Vec3b red = cv::Vec3b(255,0,0);
                //     cv::Vec3b black = cv::Vec3b(0,0,0);
                //     if(abs(cone.center_x)/10.0 < coneWidth)
                //         mat.at<cv::Vec3b>(cv::Point(i,abs(cone.center_x-parameters.input_width/2.0)/10.0))=black;
                //     mat.at<cv::Vec3b>(cv::Point(i,0))=red;
                // }
            }
            cones_cpu_.cones[i + j * coneWidth] = cone;
        }
    }

#ifdef WITH_MATPLOTLIB
    plt::figure();
    plt::named_plot("Cone pixel index 1/8th", description_map["pixel_index_x"], description_map["pixel_index"]);
    plt::title("Cone pixel index");
    plt::legend();
    //
    plt::figure();
    plt::named_plot("Cone index 1/2th", description_map["cone_index_at_half_x"], description_map["cone_index_at_half"]);
    plt::title("Cone index");
    plt::legend();

    plt::figure();
    plt::named_plot("Cone index 1/2th", description_map["cone_index_at_half_deg"], description_map["cone_index_at_half"]);
    plt::title("Cone index (deg)");
    plt::legend();

#endif

    //cv::imshow("PhotoSampling",mat);
    initPhotoGpu();
}

cv::Mat RetinaCuda::drawConeMap()
{

    cv::Mat output_mapping(cones_cpu_.height, cones_cpu_.width, CV_8UC3, cv::Vec3b(0, 0, 0));

    for (int y = 0; y < cones_cpu_.height; y++)
    {
        for (int x = 0; x < cones_cpu_.width; x++)
        {
            const auto &cone = cones_cpu_.cones[x + y * cones_cpu_.width];
            cv::Vec3b &color = output_mapping.at<cv::Vec3b>({x, y});
            switch (cone.type)
            {
            case PHOTO_TYPE::S_CONE:
                color[0] = 255;
                break;
            case PHOTO_TYPE::M_CONE:
                color[1] = 255;
                break;
            case PHOTO_TYPE::L_CONE:
                color[2] = 255;
                break;
            default:
                break;
            }
        }
    }
    return output_mapping;
}

void RetinaCuda::applyPhotoreceptorSampling(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst)
{
    imgDst.create(cones_cpu_.height, cones_cpu_.width, CV_8UC1);
    imgDst.setTo(0);

    if (imgSrc.channels() == 1)
    {
        gpu::photoreceptorSampling1C(imgSrc, imgDst, gpuCones, cones_cpu_.width, cones_cpu_.height, cudaStreamDefault /*(cudaStream_t)cuda_stream*/);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "Error: " << cudaGetErrorString(err) << " (you may have incorrect arch in cmake)" << std::endl;
            exit(1);
        }
    }
    else if (imgSrc.channels() == 3)
    {
        gpu::photoreceptorSampling3C(imgSrc, imgDst, gpuCones, cones_cpu_.width, cones_cpu_.height, cudaStreamDefault /*(cudaStream_t)cuda_stream*/);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "Error: " << cudaGetErrorString(err) << " (you may have incorrect arch in cmake)" << std::endl;
            exit(1);
        }
    }
    else
    {
        std::cerr << "Not implemented" << std::endl;
    }
}

bool RetinaCuda::initCellsGpu(std::vector<Ganglionar> cellsArrayCPU, int cellsArrayWidth, int cellsArrayHeight)
{
    if (cudaMalloc((void **)&gpuCells, sizeof(Ganglionar) * cellsArrayCPU.size()) != cudaError_t::cudaSuccess)
    {
        return false;
    }
    cudaMemcpy(gpuCells, cellsArrayCPU.data(), sizeof(Ganglionar) * cellsArrayCPU.size(), cudaMemcpyHostToDevice);

    this->cellsArrayWidth = cellsArrayWidth;
    this->cellsArrayHeight = cellsArrayHeight;

    return true;
}

bool RetinaCuda::initDirectiveGpu(std::vector<Point> photoSrc, std::vector<Point> photoDst)
{
    if (cudaMalloc((void **)&d_magnoMappingSrc, sizeof(Point) * photoSrc.size()) != cudaError_t::cudaSuccess)
    {
        return false;
    }
    cudaMemcpy(d_magnoMappingSrc, photoSrc.data(), sizeof(Point) * photoSrc.size(), cudaMemcpyHostToDevice);

    if (cudaMalloc((void **)&d_magnoMappingDst, sizeof(Point) * photoDst.size()) != cudaError_t::cudaSuccess)
    {
        return false;
    }
    cudaMemcpy(d_magnoMappingDst, photoDst.data(), sizeof(Point) * photoDst.size(), cudaMemcpyHostToDevice);

    this->magnoMappingSize = photoSrc.size();

    return true;
}

double RetinaCuda::setRandomSeed(int val)
{
    mt_rand.seed(val);
}

bool RetinaCuda::initPhotoGpu()
{
    if (cudaMalloc((void **)&gpuCones, sizeof(Cone) * cones_cpu_.cones.size()) != cudaError_t::cudaSuccess)
    {
        return false;
    }
    cudaMemcpy(gpuCones, cones_cpu_.cones.data(), sizeof(Cone) * cones_cpu_.cones.size(), cudaMemcpyHostToDevice);
    return true;
}

double RetinaCuda::getRandom()
{
    //std::mt19937 mt_rand(0);//Use always the same seed for regeneration

    return mt_rand() / ((double)(std::mt19937::max()));
}

void RetinaCuda::applyParvoGC(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst)
{
    //imgDst
    if (imgDst.cols != cellsArrayWidth || imgDst.rows != cellsArrayHeight)
    {
        imgDst.create(cellsArrayHeight, cellsArrayWidth, CV_8UC1);
        imgDst.setTo(0);
    }

    //qDebug()<<"SIZES:"<<imgSrc.cols<<imgSrc.rows<<imgDst.cols<<imgDst.rows;
    gpu::multiConvolve(imgSrc, imgDst, gpuCells, cellsArrayWidth, cellsArrayHeight, cudaStreamDefault);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Error: " << cudaGetErrorString(err) << " (you may have incorrect arch in cmake)" << std::endl;
        exit(1);
    }
}

void RetinaCuda::applyDirectiveGC(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst, cv::cuda::GpuMat &prevImage)
{
    //imgDst
    std::cout << "directive_height " << directive_height << "directive_width " << directive_width << std::endl;
    imgDst.create(directive_height, directive_width, CV_8UC1);
    imgDst.setTo(0);
    //qDebug()<<"SIZES:"<<imgSrc.cols<<imgSrc.rows<<imgDst.cols<<imgDst.rows;
    gpu::directionSelectiveComputation(imgSrc, imgDst, prevImage, d_magnoMappingSrc, d_magnoMappingDst, magnoMappingSize, cudaStreamDefault);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Error: " << cudaGetErrorString(err) << " (you may have incorrect arch in cmake)" << std::endl;
        exit(1);
    }
}

void RetinaCuda::sparse(cv::cuda::GpuMat &imgSrc, int bits, GpuBitArray2D &output, unsigned char min_value, unsigned char max_value)
{
    gpu::sparse(imgSrc, bits, output, min_value, max_value, cudaStreamDefault);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Error: " << cudaGetErrorString(err) << " (you may have incorrect arch in cmake)" << std::endl;
        exit(1);
    }
}

void RetinaCuda::discretise(cv::cuda::GpuMat &imgSrc, int vals, cv::cuda::GpuMat &output, unsigned char min_value, unsigned char max_value)
{
    output.create(imgSrc.rows, imgSrc.cols, CV_8UC1);
    gpu::discretise(imgSrc, vals, output, min_value, max_value, cudaStreamDefault);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Error: " << cudaGetErrorString(err) << " (you may have incorrect arch in cmake)" << std::endl;
        exit(1);
    }
}

void RetinaCuda::addKernels()
{
    //gpu::addKernel();
}
