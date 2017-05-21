#include <cuda.h>
#include <builtin_types.h>
#include <math.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include "device_functions.h"
#include "cuda_retina_kernels.cuh"

#include <opencv/cv.h>

namespace gpu{
/// Kernels ///


Cell* loadCellsArrayToGPU(Cell* cellsArrayHost, int width, int height){
    //calcul de la taille des matrices
    int size = width * height * sizeof(Cell);
    Cell* cellsArrayDevice;

    //allocation des matrices et leur remplissage
    //cudaMalloc(&cellsArrayDevice, sizeof(Cell*));
    cudaMalloc((void**) &cellsArrayDevice, size);
    cudaMemcpy(cellsArrayDevice, cellsArrayHost, size, cudaMemcpyHostToDevice);

//    for(int i=0; i< 10; i++){
//       printf("loadCellsArrayToGPU: %i\n",cellsArrayHost[i].center_x);
//    }

    return cellsArrayDevice;

    //TODO
//    int** a;

//    cudaMalloc(&a, sizeof(int*) * N));

//    int* ha[N];

//    for(int i = 0; i < N; ++i)

//           cudaMalloc(&ha[i],size));

//    cudaMemcpy(a, ha, sizeof(a), cudaMemcpyHostToDevice);
}

void unloadArray(Cell* cell){
    //destruction des matrices, désormais inutilisées
    cudaFree(cell);
}

__global__ void channelSamplingKernel(cv::cuda::PtrStepSz<uchar3> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, cv::cuda::PtrStepSz<u_char> samplingMapGPU){
    // Get our global thread ID
//    int xdst = blockIdx.x*blockDim.x +threadIdx.x;
//    int ydst = blockIdx.y*blockDim.y+threadIdx.y;

//    //int nbcols = samplingMapGPU.cols;
//    //int nbrows = samplingMapGPU.rows;
//    uchar3 v = imgSrc(ydst,xdst);
//    uchar out;
//    switch (samplingMapGPU(ydst,xdst)) {
//    case 0:
//        out = (v.x);
//        break;
//    case 1:
//        out = (v.y);
//        break;
//    case 2:
//        out =  (v.z);
//        break;
//    default:
//        break;
//    }

    //cv::cudev::VecTraits<uchar3>::make()
    //imgDst(ydst,xdst) = out;
    //imgDst(ydst,xdst)= imgSrc(ydst,xdst);

}

__global__ void multiConvolveKernel(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Cell* cellsArray, int cellsArrayWidth, int cellsArrayHeight){
    // Get our global thread ID
    int xdst = blockIdx.x*blockDim.x +threadIdx.x;
    int ydst = blockIdx.y*blockDim.y+threadIdx.y;

    int nbcols = imgSrc.cols;
    int nbrows = imgSrc.rows;


    Cell cell = cellsArray[(xdst+ydst*cellsArrayWidth)];//%(cellsArrayWidth*cellsArrayHeight)/*(xdst+ydst*cellsArrayWidth)%(cellsArrayWidth*cellsArrayHeight)*/];

//    if(xdst+ydst*cellsArrayWidth < 50){
//        printf(" cell %i: centerx%i \n", xdst+ydst*cellsArrayWidth, cell.center_x);

//    }



    int x = cell.center_x;
    int y = cell.center_y;

    int in_radius_squarred = cell.intern_radius*cell.intern_radius;
    int ex_radius_squarred = cell.extern_radius*cell.extern_radius;

    int xi;
    int yi;

    int value_center = 0;
    int value_ext = 0;

    int nbCenter = 0;
    int nbOut = 0;

    for(xi=-cell.extern_radius; xi <= cell.extern_radius; xi++){
        for(yi=-cell.extern_radius; yi <= cell.extern_radius; yi++){
            if(x+xi>0 && x+xi<nbcols && y+yi>0 && y+yi<nbrows){
                if(xi*xi + yi*yi < in_radius_squarred){//if we are in the radius
                    if(cell.type == 0){
                        value_center += imgSrc(y+yi,x+xi);
                    }else{
                        value_center -= imgSrc(y+yi,x+xi);
                    }
                    nbCenter++;
                }else if(xi*xi + yi*yi < ex_radius_squarred){
                    if(cell.type == 0){
                        value_ext -= imgSrc(y+yi,x+xi);
                    }else{
                        value_ext += imgSrc(y+yi,x+xi);
                    }
                    nbOut++;
                }
            }
        }
    }

    if(nbCenter == 0){
        nbCenter = 1;
    }
    if(nbOut == 0){
        nbOut = 1;
    }

    int total_value = 128 + (value_center/(float)nbCenter + value_ext/(float)nbOut)/2;


//    float sub = (value_center/(float)nbOn + value_ext/(float)nbOff)/2.0;;
//    if(xdst ==42 && 42 ==ydst){
//        printf(" cell value%i %i %i: %f\n", total_value,nbOn,value_center,sub);
//    }
    if(total_value<0){
        total_value = 0;
    }else if(total_value>255){
        total_value= 255;
    }

    if(xdst<imgDst.cols && ydst < imgDst.rows)
        imgDst(ydst,xdst)= total_value;

}

///////////////////


void channelSampling(cv::cuda::PtrStepSz<uchar3> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, cv::cuda::PtrStepSz<u_char> samplingMapGPU, cudaStream_t stream){
    //int blockSize, gridSize;
    dim3 grid, block;

    // Number of threads in each thread block
//    blockSize = 960;
    block.x = 16;
    block.y = 16;

    // Number of thread blocks in grid
    //gridSize = (int)ceil((float)(imgDst.cols*imgDst.rows)/blockSize);
    grid.x = (int)ceil((float)(imgDst.cols)/block.x);
    grid.y = (int)ceil((float)(imgDst.rows)/block.y);


    channelSamplingKernel<<<grid, block>>>(imgSrc, imgDst, samplingMapGPU);
}

void multiConvolve(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Cell* cellsArrayGPU, int cellsArrayWidth, int cellsArrayHeight,cudaStream_t stream){


    //int blockSize, gridSize;
    dim3 grid, block;

    // Number of threads in each thread block
//    blockSize = 960;
    block.x = 16;
    block.y = 16;

    // Number of thread blocks in grid
    //gridSize = (int)ceil((float)(imgDst.cols*imgDst.rows)/blockSize);
    grid.x = (int)ceil((float)(imgDst.cols)/block.x);
    grid.y = (int)ceil((float)(imgDst.rows)/block.y);

    multiConvolveKernel<<<grid, block>>>(imgSrc, imgDst, cellsArrayGPU, cellsArrayWidth,cellsArrayHeight);

}

}
