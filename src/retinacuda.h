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
    RetinaCuda(int gpu = 0);
    void initRetina(int cellsArrayWidth, int cellsArrayHeight);
    void applyPhotoreceptorSampling(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst, cudaStream_t cuda_stream = cudaStreamDefault);
    void applyMultiConvolve(cv::cuda::GpuMat &imgSrc, cv::cuda::GpuMat &imgDst, cudaStream_t stream = cudaStreamDefault);

//Test
    void addKernels();

protected:
    virtual std::vector<Cell> initGanglionarCells(int cellsArrayWidth, int cellsArrayHeight);
    virtual cv::Mat initConeSampling(int cellsArrayWidth, int cellsArrayHeight);

private:
    bool initCellsArray(std::vector<Cell> cellsArrayCPU, int cellsArrayWidth, int cellsArrayHeight);
    double getRandom();
    Cell * gpuCells;
    cv::cuda::GpuMat gpuChannelSampling;
    int cellsArrayWidth;
    int cellsArrayHeight;
};

#endif // RETINACUDA_H
