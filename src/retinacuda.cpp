#include "retinacuda.h"
#include <opencv2/core/cuda_types.hpp>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda.hpp>
#include "Cuda/cuda_image.h"

#include "Cuda/cuda_retina_kernels.cuh"

//Human mgc
float excentricity(float r){
    float dgf0 = 30000; // density at r0
    float ak = 0.9729; // first term weight
    float r2k = 1.084;
    float rek = 7.633;

    //http://jov.arvojournals.org/article.aspx?articleid=2279458#87788067
    return dgf0 * (ak*pow((1 + r/r2k),-2) + (1-ak)*exp(-r/rek));
}

class RetinaCudaException : public std::exception{
public:
    RetinaCudaException(std::string message){
        this->message = message;
    }


    // exception interface
public:
    const char *what() const throw(){
        return message.c_str();
    }

private:
    std::string message;
};

RetinaCuda::RetinaCuda(int gpu)
{
    gpuCells = 0;

    int count = cv::cuda::getCudaEnabledDeviceCount();

    if(count == 0){
        throw RetinaCudaException("No gpu avaliable");
    }else if(count<gpu){
        throw RetinaCudaException("Gpu not avaliable");
    }
    cv::cuda::setDevice(gpu);

    cv::cuda::DeviceInfo deviceInfos(gpu);

    std::cout<<"Using gpu: "<<deviceInfos.name()<<std::endl;
}

void RetinaCuda::initRetina(Parameters param)
{

    //cellsArrayWidth = param.ganglionar_cells_width;
    //cellsArrayHeight = param.ganglionar_cells_height;
    parameters = param;
    std::vector<Cone> coneMapping = initCone(param.input_width, param.input_height);
    std::vector<Ganglionar> cells = initGanglionarCells(conesArrayWidth,conesArrayHeight);


    //gpuChannelSampling.create(cellsArrayWidth,cellsArrayHeight,1);
    //gpuChannelSampling.upload(coneMapping);

    //initCellsArray(cells,cellsArrayWidth,cellsArrayHeight);
}

std::vector<Ganglionar> RetinaCuda::initGanglionarCells(int conesWidth, int conesHeight)
{
    std::vector<Ganglionar> cellsCPU;

    //Findout cellWidth and cellHeight
    int cellsWidth = -1;
    int cellsHeight = -1;

    //Get cone width
    for(int r=0; r<INT_MAX/2 && cellsWidth < 0;r++){
        cv::Vec2f direction(1,0);
        cv::Point src_pos =  getPosition(mgc_position_mapping(r),cv::Size(conesWidth,conesHeight),direction);
        if(src_pos.x >= conesWidth || src_pos.y >= conesHeight ||
                src_pos.x < 0 || src_pos.y < 0){
            cellsWidth = (r-1)*2;
            cellsWidth = cellsWidth - cellsWidth%16;
        }
    }
    //Get cone height
    for(int r=0; r<INT_MAX/2 && cellsHeight<0;r++){
        cv::Vec2f direction(0,1);
        cv::Point src_pos =  getPosition(mgc_position_mapping(r),cv::Size(conesWidth,conesHeight),direction);
        if(src_pos.x >= conesWidth || src_pos.y >= conesHeight ||
                src_pos.x < 0 || src_pos.y < 0){
            cellsHeight = (r-1)*2;
            cellsHeight = cellsHeight - cellsHeight%16;
        }
    }


    if(cellsHeight <= 0 || cellsWidth <= 0){
        std::cerr<<"Parameter implies empty cone array"<<std::endl;
        return cellsCPU;
    }

    cellsArrayHeight = cellsHeight;
    cellsArrayWidth = cellsWidth;

    cellsCPU.resize(cellsArrayWidth*cellsArrayHeight);

    Ganglionar cell;
    double r;
    double ganglionarExternalRadius;
    double ganglionarInternalRadius;

    cv::Mat mat(cellsArrayHeight,cellsArrayWidth,CV_8UC3,cv::Vec3b(255,255,255));

    //Default model
    for(int j=0; j<cellsArrayHeight;j++){
        for(int i=0; i<cellsArrayWidth;i++){
            //int linearReduction = 6;
            r = sqrt((cellsArrayWidth/2.0-i)*(cellsArrayWidth/2.0-i)+(cellsArrayHeight/2.0-j)*(cellsArrayHeight/2.0-j));
            ganglionarExternalRadius = mgc_dentric_coverage(r);
            ganglionarInternalRadius = MAX(1.0,ganglionarExternalRadius/2.0);
            cv::Vec2f direction = getDirectionFromCenter(cv::Point(i,j),cv::Size(cellsArrayWidth,cellsArrayHeight));
            cv::Point src_pos = getPosition(mgc_position_mapping(r),cv::Size(conesArrayWidth,conesArrayHeight) ,direction);
            cell.center_x = src_pos.x;
            cell.center_y = src_pos.y;
            cell.extern_radius = ganglionarExternalRadius;
            cell.intern_radius = ganglionarInternalRadius;
            cell.type = getRandom() < 0.5 ? 0: 1;
            cellsCPU[i+j*cellsArrayWidth] = cell;
            //memcpy( cellsArrayGPU + i+j*cellsArrayWidth,&cell, sizeof(cell));
            //cellsArrayGPU[i+j*cellsArrayWidth] = cell;

            //Display stuff
            if(j == cellsArrayHeight/2){
                cv::Vec3b red = cv::Vec3b(255,0,0);
                cv::Vec3b black = cv::Vec3b(0,0,0);
                mat.at<cv::Vec3b>(cv::Point(i,j+ganglionarInternalRadius))=black;
                mat.at<cv::Vec3b>(cv::Point(i,j+ganglionarExternalRadius))=cv::Vec3b(0,255,0);
                mat.at<cv::Vec3b>(cv::Point(i,j))=red;
            }
        }
    }

    cv::imshow("Cells",mat);
    initCellsArray(cellsCPU,cellsArrayWidth,cellsArrayHeight);
    //free(cellsArrayGPU);
    return cellsCPU;
}

std::vector<Cone> RetinaCuda::initCone(int inputWidth, int inputHeight)
{
    std::vector<Cone> conesCPU;

    //Findout coneWidth and coneHeight
    int coneWidth = -1;
    int coneHeight = -1;

    //Get cone width
    for(int r=0; r<INT_MAX/2 && coneWidth < 0;r++){
        cv::Vec2f direction(1,0);
        cv::Point src_pos =  getPosition(cone_distance_mapping(r),cv::Size(inputWidth,inputHeight),direction);
        if(src_pos.x >= inputWidth || src_pos.y >= inputHeight ||
                src_pos.x < 0 || src_pos.y < 0){
            coneWidth = (r-1)*2;
            coneWidth = coneWidth - coneWidth%16;
        }
    }
    //Get cone height
    for(int r=0; r<INT_MAX/2 && coneHeight<0;r++){
        cv::Vec2f direction(0,1);
        cv::Point src_pos =  getPosition(cone_distance_mapping(r),cv::Size(inputWidth,inputHeight),direction);
        if(src_pos.x >= inputWidth || src_pos.y >= inputHeight ||
                src_pos.x < 0 || src_pos.y < 0){
            coneHeight = (r-1)*2;
            coneHeight = coneHeight - coneHeight%16;
        }
    }

    if(coneHeight <= 0 || coneWidth <= 0){
        std::cerr<<"Parameter implies empty cone array"<<std::endl;
        return conesCPU;
    }


    conesCPU.resize(coneWidth*coneHeight);

    Cone cone;
    double r;

    cv::Mat mat(coneWidth,coneWidth,CV_8UC3,cv::Vec3b(255,255,255));

    //Default model
    for(int j=0; j<coneHeight;j++){
        for(int i=0; i<coneWidth;i++){
            //int linearReduction = 6;
            r = sqrt((coneWidth/2.0-i)*(coneWidth/2.0-i)+(coneHeight/2.0-j)*(coneHeight/2.0-j));
            //ganglionarExternalRadius = cone_coverage(r);
            //ganglionarInternalRadius = MAX(1.0,ganglionarExternalRadius/2.0);
            cv::Vec2f direction = getDirectionFromCenter(cv::Point(i,j),cv::Size(coneWidth,coneHeight));
            cv::Point src_pos =  getPosition(cone_distance_mapping(r),cv::Size(parameters.input_width,parameters.input_height),direction);

            if(src_pos.x >= parameters.input_width || src_pos.y >= parameters.input_height ||
                    src_pos.x < 0 || src_pos.y < 0){
                cone.type = -1;
            }else{

                cone.center_x = src_pos.x;
                cone.center_y = src_pos.y;
                cone.type = ((int)(getRandom() * 4.0)) % 3;

                //Display stuff
                if(j == coneHeight/2){
                    cv::Vec3b red = cv::Vec3b(255,0,0);
                    cv::Vec3b black = cv::Vec3b(0,0,0);
                    if(abs(cone.center_x-i) < coneWidth)
                        mat.at<cv::Vec3b>(cv::Point(i,abs(cone.center_x-i)))=black;
                    mat.at<cv::Vec3b>(cv::Point(i,j))=red;
                }
            }

            conesCPU[i+j*coneWidth] = cone;
            //memcpy( cellsArrayGPU + i+j*cellsArrayWidth,&cell, sizeof(cell));
            //cellsArrayGPU[i+j*cellsArrayWidth] = cell;


        }
    }

    cv::imshow("PhotoSampling",mat);
    initPhotoArray(conesCPU,coneWidth,coneHeight);
    //free(cellsArrayGPU);
    return conesCPU;
}



void RetinaCuda::applyPhotoreceptorSampling(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst, cudaStream_t cuda_stream)
{
    imgDst.create(conesArrayHeight,conesArrayWidth,CV_8UC1);

    //gpu::channelSampling(imgSrc,imgDst,gpuChannelSampling,cuda_stream);

    if(imgSrc.channels() == 1){
        gpu::photoreceptorSampling1C(imgSrc,imgDst,gpuCones,conesArrayWidth,conesArrayHeight,cuda_stream);
    }else{
        std::cerr<<"Not implemented"<<std::endl;
    }

}

bool RetinaCuda::initCellsArray(std::vector<Ganglionar> cellsArrayCPU, int cellsArrayWidth, int cellsArrayHeight)
{
    if(cudaMalloc((void **) &gpuCells, sizeof(Ganglionar)*cellsArrayCPU.size()) != cudaError_t::cudaSuccess){
        return false;
    }
    cudaMemcpy(gpuCells,cellsArrayCPU.data(),sizeof(Ganglionar)*cellsArrayCPU.size(),cudaMemcpyHostToDevice);


    this->cellsArrayWidth = cellsArrayWidth;
    this->cellsArrayHeight = cellsArrayHeight;

    return true;
}

bool RetinaCuda::initPhotoArray(std::vector<Cone> conesArrayCPU, int conesArrayWidth, int conesArrayHeight)
{
    if(cudaMalloc((void **) &gpuCones, sizeof(Cone)*conesArrayCPU.size()) != cudaError_t::cudaSuccess){
        return false;
    }
    cudaMemcpy(gpuCones,conesArrayCPU.data(),sizeof(Cone)*conesArrayCPU.size(),cudaMemcpyHostToDevice);


    this->conesArrayWidth = conesArrayWidth;
    this->conesArrayHeight = conesArrayHeight;

    return true;
}


double RetinaCuda::getRandom()
{
    return rand() / ((double)(RAND_MAX));
}

void RetinaCuda::applyMultiConvolve(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst, cudaStream_t stream)
{
    //imgDst
    imgDst.create(cellsArrayHeight,cellsArrayWidth,CV_8UC1);
    //qDebug()<<"SIZES:"<<imgSrc.cols<<imgSrc.rows<<imgDst.cols<<imgDst.rows;
    gpu::multiConvolve(imgSrc,imgDst,gpuCells,cellsArrayWidth,cellsArrayHeight,stream);
}

void RetinaCuda::applySelectiveGC(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst, cv::cuda::GpuMat &prevImage, cudaStream_t stream)
{
    //imgDst
    imgDst.create(conesArrayHeight,conesArrayWidth,CV_8UC1);
    //qDebug()<<"SIZES:"<<imgSrc.cols<<imgSrc.rows<<imgDst.cols<<imgDst.rows;
    gpu::directionSelectiveComputation(imgSrc,imgDst,prevImage,stream);
}

void RetinaCuda::sparse(cv::cuda::GpuMat &imgSrc, int bits, GpuBitArray2D &output, unsigned char min_value, unsigned char max_value, cudaStream_t stream)
{
    gpu::sparse(imgSrc,bits,output,min_value,max_value,stream);
}

void RetinaCuda::discretise(cv::cuda::GpuMat &imgSrc, int vals, cv::cuda::GpuMat &output, unsigned char min_value, unsigned char max_value, cudaStream_t stream)
{
    output.create(imgSrc.rows,imgSrc.cols,CV_8UC1);
    gpu::discretise(imgSrc,vals,output,min_value,max_value,stream);
}

void RetinaCuda::addKernels()
{
    //gpu::addKernel();
}


