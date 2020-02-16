#ifndef RETINACUDA_H
#define RETINACUDA_H

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/cuda.hpp"
#include "Cuda/retinastructs.h"
//#include <cuda.h>
//#include <builtin_types.h>
#include <random>
#include "Cuda/cuda_arrays.h"

class RetinaCuda
{
public:

    struct Parameters{
        Parameters(){
            input_width = 0;
            input_height = 0;
            random_seed = 0;
        }
        //Raw input params
        int input_width;
        int input_height;

        //Photoreceptor params
        ////int cones_width;
        //int cones_height;
        double ph_fovea_radius;
        double ph_fovea_pixels_by_cone;
        double ph_max_pixels_by_cone;
        double ph_max_pixels_by_cone_radius;

        //Ganglionar params
        //int ganglionar_cells_width;
        //int ganglionar_cells_height;
        double gc_fovea_radius;
        double gc_fovea_cones_by_cell;
        double gc_fovea_inter_cells_distance;
        double gc_max_cones_by_cell;
        double gc_max_cones_by_cell_radius;
        double gc_max_inter_cells_distance;

        int random_seed;
    };

    RetinaCuda(int gpu = 0);
    ~RetinaCuda();
    void initRetina(Parameters param);

    ///
    /// \brief applyPhotoreceptorSampling
    /// \param imgSrc
    /// \param imgDst
    ///
    void applyPhotoreceptorSampling(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst);

    ///
    /// \brief
    /// \param imgSrc
    /// \param imgDst
    ///
    void applyParvoGC(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst);

    ///
    /// \brief
    /// \param imgSrc
    /// \param imgDst
    /// \param prevImage
    ///
    void applyDirectiveGC(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst, cv::cuda::GpuMat &prevImage);

    //
    void sparse(cv::cuda::GpuMat &imgSrc, int bits, GpuBitArray2D &output, unsigned char min_value = 0,unsigned char max_value = 255);

    //Test
    void discretise(cv::cuda::GpuMat &imgSrc, int vals, cv::cuda::GpuMat &output, unsigned char min_value = 0, unsigned char max_value = 255);
    void addKernels();

protected:
    virtual std::vector<Cone> initCone(int inputWidth, int inputHeight);
    virtual std::vector<Ganglionar> initGanglionarCells(int conesWidth, int conesHeight);
    virtual std::vector<Point> initSelectiveCells();
   // virtual cv::Mat initConeSampling(int cellsArrayWidth, int cellsArrayHeight);

private:
    bool initPhotoGpu(std::vector<Cone> conesArrayCPU, int conesArrayWidth, int conesArrayHeight);
    bool initCellsGpu(std::vector<Ganglionar> cellsArrayCPU, int cellsArrayWidth, int cellsArrayHeight);
    bool initDirectiveGpu(std::vector<Point> photoSrc, std::vector<Point> photoDst);

    /// Generation functions ///
    double setRandomSeed(int val);
    double getRandom();
    double mgc_dentric_coverage(float distance_from_center){
            // 2 -> 0.5 -> 0.05
        if(distance_from_center <= parameters.gc_fovea_radius){
            return parameters.gc_fovea_cones_by_cell;
        }else{
            double coeff;
            if(distance_from_center < parameters.gc_max_cones_by_cell_radius){
                coeff = ((distance_from_center-parameters.gc_fovea_radius)/(parameters.gc_max_cones_by_cell_radius-parameters.gc_fovea_radius));
            }else{
                coeff = 1;
            }

            return (parameters.gc_fovea_cones_by_cell+(parameters.gc_max_cones_by_cell-parameters.gc_fovea_cones_by_cell)*coeff);
        }
    }

    double mgc_position_mapping (double distance_from_center){

        if(distance_from_center < parameters.gc_fovea_radius){
            return distance_from_center*parameters.gc_fovea_inter_cells_distance;
        }else{
            double fovea_offset = parameters.gc_fovea_radius*parameters.gc_fovea_inter_cells_distance;
            double distance_from_fovea = distance_from_center-parameters.gc_fovea_radius;

            double coeff;
            if(distance_from_center < parameters.gc_max_cones_by_cell_radius){
                coeff = ((distance_from_center-parameters.gc_fovea_radius)/(parameters.gc_max_cones_by_cell_radius-parameters.gc_fovea_radius));
            }else{
                coeff = 1;
            }
            //double coeff = parameters.fovea_cells_by_pixel*(parameters.min_cells_by_pixel-parameters.fovea_cells_by_pixel);
            double step_size = parameters.gc_fovea_inter_cells_distance+(parameters.gc_max_inter_cells_distance-parameters.gc_fovea_inter_cells_distance)*coeff;
            return fovea_offset +distance_from_fovea*step_size;
        }
    }

    double cone_distance_mapping (double cone_distance_from_center){

        if(cone_distance_from_center <= parameters.ph_fovea_radius){
            return cone_distance_from_center*parameters.ph_fovea_pixels_by_cone;
        }else{
            double fovea_offset = parameters.ph_fovea_radius*parameters.ph_fovea_pixels_by_cone;
            double distance_from_fovea = cone_distance_from_center-parameters.ph_fovea_radius;
            double coeff;

             if(cone_distance_from_center < parameters.ph_max_pixels_by_cone_radius ){
                 coeff = ((cone_distance_from_center-parameters.ph_fovea_radius)/(parameters.ph_max_pixels_by_cone_radius-parameters.ph_fovea_radius));
             }else{
                 coeff = 1;
             }
            //double coeff = parameters.fovea_cells_by_pixel*(parameters.min_cells_by_pixel-parameters.fovea_cells_by_pixel);
            double step_size = parameters.ph_fovea_pixels_by_cone+(parameters.ph_max_pixels_by_cone-parameters.ph_fovea_pixels_by_cone)*coeff;
            return fovea_offset +distance_from_fovea*step_size;
        }
    }

//    double getConesMaxRadius(){
//        return sqrt((double)parameters*parameters.cones_width + parameters.cones_height*parameters.cones_height)/2;
//    }

    cv::Point convertPosToCenter(int posx, int posy, int max_x, int max_y){
        cv::Point(posx-max_x/2.0,posy-max_y/2.0);
    }

    cv::Vec2f getDirectionFromCenter(cv::Point topLeftPosition,cv::Size size){
        return cv::Vec2f(topLeftPosition.x-size.width/2.0,topLeftPosition.y-size.height/2.0 );
    }

    cv::Point getPosition(double a_distance_from_center, cv::Size a_size, cv::Vec2f direction){
        cv::Vec2f normalizedDirection = cv::normalize(direction);
        double x_center = a_distance_from_center*normalizedDirection[0];
        double y_center = a_distance_from_center*normalizedDirection[1];

        return cv::Point(round(x_center+a_size.width/2.0),round(y_center+a_size.height/2.0));
    }

//    cv::Point getPosition(double src_distance, int magno_x, int magno_y){
//        double magno_center_x = magno_x-cellsArrayWidth/2.0;
//        double magno_center_y = magno_y-cellsArrayHeight/2.0;
//        double magno_distance = sqrt(magno_center_x*magno_center_x+magno_center_y*magno_center_y);
//        double x_center = src_distance*magno_center_x/magno_distance;
//        double y_center = src_distance*magno_center_y/magno_distance;
//        return cv::Point(x_center+parameters.input_width/2.0,y_center+parameters.input_height/2.0);
//    }


    Cone * gpuCones;
    Ganglionar * gpuCells;
    Point * d_magnoMappingSrc;
    Point * d_magnoMappingDst;
    int directive_width;
    int directive_height;
    int magnoMappingSize;
    cv::cuda::GpuMat gpuChannelSampling;
    std::vector<Point> magnoMappingSrc;
    std::vector<Point> magnoMappingDst;
    int cellsArrayWidth;
    int cellsArrayHeight;
    int conesArrayWidth;
    int conesArrayHeight;
    Parameters parameters;
    std::vector<int> coneMarge;
    //std::shared_ptr<void *> cuda_stream_ptr;

    std::mt19937 mt_rand;

};

#endif // RETINACUDA_H
