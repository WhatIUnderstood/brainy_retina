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
//extern "C"{
//void addKernel();
//void multiConvolve(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Cell* cellsArrayGPU, int cellsArrayWidth, int cellsArrayHeight);
//void channelSampling(cv::cuda::PtrStepSz<uchar3> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, cv::cuda::PtrStepSz<u_char> samplingMapGPU);

//Cell* loadCellsArrayToGPU(Cell* cellsArrayHost, int width, int height);

//}

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

void RetinaCuda::initRetina(int cellsArrayWidth, int cellsArrayHeight)
{

    cv::Mat coneMapping = initConeSampling(cellsArrayWidth, cellsArrayHeight);
    std::vector<Cell> cells = initGanglionarCells(cellsArrayWidth,cellsArrayHeight);


    gpuChannelSampling.create(cellsArrayWidth,cellsArrayHeight,1);
    gpuChannelSampling.upload(coneMapping);

    //initCellsArray(cells,cellsArrayWidth,cellsArrayHeight);
}

std::vector<Cell> RetinaCuda::initGanglionarCells(int cellsArrayWidth, int cellsArrayHeight)
{
    std::vector<Cell> cellsCPU;
    cellsCPU.resize(cellsArrayWidth*cellsArrayHeight);

    //Cell *cellsArrayGPU = ((Cell*)(malloc(sizeof(Cell)*cellsArrayWidth*cellsArrayHeight)));

    Cell cell;
    double r;
    double ganglionarExternalRadius;
    double ganglionarInternalRadius;

    cv::Mat mat(cellsArrayHeight,cellsArrayWidth,CV_8UC3,cv::Vec3b(255,255,255));

    //Default model
    for(int j=0; j<cellsArrayHeight;j++){
        for(int i=0; i<cellsArrayWidth;i++){
            int linearReduction = 6;
            r = sqrt((cellsArrayWidth/2.0-i)*(cellsArrayWidth/2.0-i)+(cellsArrayHeight/2.0-j)*(cellsArrayHeight/2.0-j));
            ganglionarExternalRadius = MIN(MAX(2.0,(r/linearReduction + (getRandom()-1/2.0)*(r/linearReduction/2.0))),60.0);
            ganglionarInternalRadius = MAX(1.0,ganglionarExternalRadius/2.0);
            cell.center_x = i;
            cell.center_y = j;
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

//    //Default model
//    for(int j=0; j<cellsArrayHeight;j++){
//        for(int i=0; i<cellsArrayWidth;i++){
//            int linearReduction = 6;
//            r = sqrt((cellsArrayWidth/2.0-i)*(cellsArrayWidth/2.0-i)+(cellsArrayHeight/2.0-j)*(cellsArrayHeight/2.0-j));
//            ganglionarExternalRadius = MIN(MAX(2.0,(r/linearReduction + (getRandom()-1/2.0)*(r/linearReduction/2.0))),60.0);
//            ganglionarInternalRadius = MAX(1.0,ganglionarExternalRadius/2.0);
//            cell.center_x = i;
//            cell.center_y = j;
//            cell.extern_radius = ganglionarExternalRadius;
//            cell.intern_radius = ganglionarInternalRadius;
//            cell.type = getRandom() < 0.5 ? 0: 1;
//            cellsCPU[i+j*cellsArrayWidth] = cell;
//            //memcpy( cellsArrayGPU + i+j*cellsArrayWidth,&cell, sizeof(cell));
//            //cellsArrayGPU[i+j*cellsArrayWidth] = cell;

//            //Display stuff
//            if(j == cellsArrayHeight/2){
//                cv::Vec3b red = cv::Vec3b(255,0,0);
//                cv::Vec3b black = cv::Vec3b(0,0,0);
//                mat.at<cv::Vec3b>(cv::Point(i,j+ganglionarInternalRadius))=black;
//                mat.at<cv::Vec3b>(cv::Point(i,j+ganglionarExternalRadius))=cv::Vec3b(0,255,0);
//                mat.at<cv::Vec3b>(cv::Point(i,j))=red;
//            }
//        }
//    }

    cv::imshow("Cells",mat);
    initCellsArray(cellsCPU,cellsArrayWidth,cellsArrayHeight);
    //free(cellsArrayGPU);
    return cellsCPU;
}

cv::Mat RetinaCuda::initConeSampling(int cellsArrayWidth, int cellsArrayHeight)
{
    cv::Mat mat(cellsArrayWidth,cellsArrayHeight,CV_8UC1);

    return mat;
}

void RetinaCuda::applyPhotoreceptorSampling(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst, cudaStream_t cuda_stream)
{
    imgDst.create(imgSrc.rows,imgSrc.cols,CV_8UC1);
    gpu::channelSampling(imgSrc,imgDst,gpuChannelSampling,cuda_stream);
}

bool RetinaCuda::initCellsArray(std::vector<Cell> cellsArrayCPU, int cellsArrayWidth, int cellsArrayHeight)
{
    //cudaMalloc(0, 5);
    //cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice) ;
    ///std::cout<<"LOADING CELLS DESCRIPTIONS "<<cellsArrayWidth*cellsArrayHeight<<std::endl;

//    Cell *cellsArrayCPUMem = ((Cell*)(malloc(sizeof(Cell)*cellsArrayWidth*cellsArrayHeight)));

//    for(int i=0; i<cellsArrayWidth*cellsArrayHeight;i++){
//        cellsArrayCPUMem[i]=cellsArrayCPU[i];
//    }

    if(cudaMalloc((void **) &gpuCells, sizeof(Cell)*cellsArrayCPU.size()) != cudaError_t::cudaSuccess){
        return false;
    }
    cudaMemcpy(gpuCells,cellsArrayCPU.data(),sizeof(Cell)*cellsArrayCPU.size(),cudaMemcpyHostToDevice);

    //free(cellsArrayCPUMem);

    this->cellsArrayWidth = cellsArrayWidth;
    this->cellsArrayHeight = cellsArrayHeight;

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

void RetinaCuda::addKernels()
{
    //gpu::addKernel();
}
