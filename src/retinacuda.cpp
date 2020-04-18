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

#include "layers/ConeLayer.h"
#include "layers/MGCellLayer.h"

#ifdef WITH_MATPLOTLIB
#include "matplotlib_cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

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

    int count;
    cudaError_t err = cudaGetDeviceCount(&count);

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

    MGCellsSim::MGCellsConfig mgc_config;
    mgc_config.density_params.a = 0.9729;
    mgc_config.density_params.r2 = 1.084;
    mgc_config.density_params.re = 7.633;
    mgc_config.density_params.rm = 41.03;
    mgc_config.density_params.max_cone_density = 14804;

    std::unique_ptr<ConeModel> cone_model_ptr = std::make_unique<ConeModel>(parameters.ph_config);
    std::unique_ptr<PixelConeModel> pixel_model_ptr = std::make_unique<PixelConeModel>(parameters.pix_config);
    std::unique_ptr<MGCellsSim> mgc_model_ptr = std::make_unique<MGCellsSim>(mgc_config);

    cone_layer_ptr_ = std::make_unique<ConeLayer>(cone_model_ptr, pixel_model_ptr, parameters.random_seed);
    mgcells_layer_ptr_ = std::make_unique<MGCellLayer>(mgc_model_ptr, *cone_layer_ptr_, parameters.random_seed);

    initPhotoGpu(cone_layer_ptr_->cones());
    initCellsGpu(mgcells_layer_ptr_->mgcells());

    initSelectiveCells();
}

std::vector<Point> RetinaCuda::initSelectiveCells()
{
    magnoMappingSrc.clear();
    magnoMappingDst.clear();
    const Cones &cones_cpu = cone_layer_ptr_->cones();
    // int x = 0;
    // int y = 0;
    // int max_x = 0;

    unsigned int width = cones_cpu.width / 2;
    width -= width % BLOCK_SIZE;
    unsigned int height = cones_cpu.height / 2;
    height -= height % BLOCK_SIZE;
    for (unsigned int h = 0; h < height; h++)
    {
        //x = 0;
        for (unsigned int w = 0; w < width; w++)
        {
            magnoMappingSrc.push_back(Point(w * 2, h * 2));
            magnoMappingDst.push_back(Point(w, h));
        }
        //y++;
    }
    directive_width = width;
    directive_height = height;
    initDirectiveGpu(magnoMappingSrc, magnoMappingDst);

    return magnoMappingSrc;
}

cv::Mat RetinaCuda::drawConeMap()
{

    const Cones &cones_cpu = cone_layer_ptr_->cones();
    cv::Mat output_mapping(cones_cpu.height, cones_cpu.width, CV_8UC3, cv::Vec3b(0, 0, 0));

    for (int y = 0; y < cones_cpu.height; y++)
    {
        for (int x = 0; x < cones_cpu.width; x++)
        {
            const auto &cone = cones_cpu.cones[x + y * cones_cpu.width];
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
    const Cones &cones_cpu = cone_layer_ptr_->cones();
    imgDst.create(cones_cpu.height, cones_cpu.width, CV_8UC1);
    imgDst.setTo(0);

    if (imgSrc.channels() == 1)
    {
        gpu::photoreceptorSampling1C(imgSrc, imgDst, gpuCones, cones_cpu.width, cones_cpu.height, cudaStreamDefault /*(cudaStream_t)cuda_stream*/);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "Error: " << cudaGetErrorString(err) << " (you may have incorrect arch in cmake)" << std::endl;
            exit(1);
        }
    }
    else if (imgSrc.channels() == 3)
    {
        gpu::photoreceptorSampling3C(imgSrc, imgDst, gpuCones, cones_cpu.width, cones_cpu.height, cudaStreamDefault /*(cudaStream_t)cuda_stream*/);
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

bool RetinaCuda::initCellsGpu(const GanglionarCells &mgcells)
{
    if (cudaMalloc((void **)&gpuCells, sizeof(Ganglionar) * mgcells.gcells.size()) != cudaError_t::cudaSuccess)
    {
        return false;
    }
    cudaMemcpy(gpuCells, mgcells.gcells.data(), sizeof(Ganglionar) * mgcells.gcells.size(), cudaMemcpyHostToDevice);
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

bool RetinaCuda::initPhotoGpu(const Cones &cones_cpu)
{
    if (cudaMalloc((void **)&gpuCones, sizeof(Cone) * cones_cpu.cones.size()) != cudaError_t::cudaSuccess)
    {
        return false;
    }
    cudaMemcpy(gpuCones, cones_cpu.cones.data(), sizeof(Cone) * cones_cpu.cones.size(), cudaMemcpyHostToDevice);
    return true;
}

void RetinaCuda::applyParvoGC(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst)
{
    //imgDst
    if (imgDst.cols != mgcells_layer_ptr_->mgcells().width || imgDst.rows != mgcells_layer_ptr_->mgcells().height)
    {
        imgDst.create(mgcells_layer_ptr_->mgcells().height, mgcells_layer_ptr_->mgcells().width, CV_8UC1);
        imgDst.setTo(0);
    }

    gpu::multiConvolve(imgSrc, imgDst, gpuCells, mgcells_layer_ptr_->mgcells().width, mgcells_layer_ptr_->mgcells().height, cudaStreamDefault);
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
    gpu::directionSelectiveComputation(imgSrc, imgDst, prevImage, d_magnoMappingSrc, d_magnoMappingDst, magnoMappingSize, cudaStreamDefault);
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