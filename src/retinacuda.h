#ifndef RETINACUDA_H
#define RETINACUDA_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/cuda.hpp"
#include "Cuda/retinastructs.h"
#include <cuda.h>
#include <builtin_types.h>

class RetinaCuda
{
public:

    struct Parameters{
        int cells_width;
        int cells_height;

        int input_width;
        int input_height;

        double fovea_magnocellular_radius;

        double fovea_pixels_by_cell;
        double max_pixels_by_cell;

        double fovea_inter_cells_distance;
        double max_inter_cells_distance;
    };

    RetinaCuda(int gpu = 0);
    void initRetina(Parameters param);
    void applyPhotoreceptorSampling(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst, cudaStream_t cuda_stream = cudaStreamDefault);
    void applyMultiConvolve(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst, cudaStream_t stream = cudaStreamDefault);

//Test
    void addKernels();

protected:
    virtual std::vector<Cell> initGanglionarCells();
    virtual cv::Mat initConeSampling(int cellsArrayWidth, int cellsArrayHeight);

private:
    bool initCellsArray(std::vector<Cell> cellsArrayCPU, int cellsArrayWidth, int cellsArrayHeight);

    /// Generation functions ///
    double getRandom();
    double mgc_dentric_coverage(float distance_from_center){
            // 2 -> 0.5 -> 0.05
        if(distance_from_center < parameters.fovea_magnocellular_radius){
            return parameters.fovea_pixels_by_cell;
        }else{
            double coeff = ((distance_from_center-parameters.fovea_magnocellular_radius)/(getMagnoMaxRadius()-parameters.fovea_magnocellular_radius));
            return (parameters.fovea_pixels_by_cell+(parameters.max_pixels_by_cell-parameters.fovea_pixels_by_cell)*coeff);
        }
    }

    double mgc_position_mapping (double distance_from_center){

        if(distance_from_center < parameters.fovea_magnocellular_radius){
            return distance_from_center*parameters.fovea_inter_cells_distance;
        }else{
            double fovea_offset = parameters.fovea_magnocellular_radius*parameters.fovea_inter_cells_distance;
            double distance_from_fovea = distance_from_center-parameters.fovea_magnocellular_radius;
            double coeff = ((distance_from_center-parameters.fovea_magnocellular_radius)/(getMagnoMaxRadius()-parameters.fovea_magnocellular_radius));
            //double coeff = parameters.fovea_cells_by_pixel*(parameters.min_cells_by_pixel-parameters.fovea_cells_by_pixel);
            double step_size = parameters.fovea_inter_cells_distance+(parameters.max_inter_cells_distance-parameters.fovea_inter_cells_distance)*coeff;
            return fovea_offset +distance_from_fovea*step_size;
        }
    }
    double getMagnoMaxRadius(){
        return sqrt((double)cellsArrayWidth*cellsArrayWidth + cellsArrayHeight*cellsArrayHeight)/2;
    }
    cv::Point getPosition(double src_distance, int magno_x, int magno_y){
        double magno_center_x = magno_x-cellsArrayWidth/2.0;
        double magno_center_y = magno_y-cellsArrayHeight/2.0;
        double magno_distance = sqrt(magno_center_x*magno_center_x+magno_center_y*magno_center_y);
        double x_center = src_distance*magno_center_x/magno_distance;
        double y_center = src_distance*magno_center_y/magno_distance;
        return cv::Point(x_center+parameters.input_width/2.0,y_center+parameters.input_height/2.0);
    }


    Cell * gpuCells;
    cv::cuda::GpuMat gpuChannelSampling;
    int cellsArrayWidth;
    int cellsArrayHeight;
    Parameters parameters;


};

#endif // RETINACUDA_H
