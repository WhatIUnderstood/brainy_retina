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


Ganglionar* loadCellsArrayToGPU(Ganglionar* cellsArrayHost, int width, int height){
    //calcul de la taille des matrices
    int size = width * height * sizeof(Ganglionar);
    Ganglionar* cellsArrayDevice;

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

void unloadArray(Ganglionar* cell){
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


__global__ void photoreceptorSamplingKernel1C(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Cone* conesArray, int conesArrayWidth, int /*conesArrayHeight*/){
    // Get our global thread ID
    int xdst = blockIdx.x*blockDim.x +threadIdx.x;
    int ydst = blockIdx.y*blockDim.y+threadIdx.y;

    Cone cone = conesArray[(xdst+ydst*conesArrayWidth)];//%(cellsArrayWidth*cellsArrayHeight)/*(xdst+ydst*cellsArrayWidth)%(cellsArrayWidth*cellsArrayHeight)*/];

    if(cone.type <0){
        return;
    }
    int x = cone.center_x;
    int y = cone.center_y;

//    if(xdst<imgDst.cols && ydst < imgDst.rows)
        imgDst(ydst,xdst) = imgSrc(y,x);

}

__global__ void multiConvolveKernel(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Ganglionar* cellsArray, int cellsArrayWidth, int cellsArrayHeight){
    // Get our global thread ID
    int xdst = blockIdx.x*blockDim.x +threadIdx.x;
    int ydst = blockIdx.y*blockDim.y+threadIdx.y;

    int nbcols = imgSrc.cols;
    int nbrows = imgSrc.rows;


    Ganglionar cell = cellsArray[(xdst+ydst*cellsArrayWidth)];//%(cellsArrayWidth*cellsArrayHeight)/*(xdst+ydst*cellsArrayWidth)%(cellsArrayWidth*cellsArrayHeight)*/];

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
                        value_center += (imgSrc(y+yi,x+xi)-128);
                    }else{
                        value_center -= (imgSrc(y+yi,x+xi)-128);
                    }
                    nbCenter++;
                }else if(xi*xi + yi*yi < ex_radius_squarred){
                    if(cell.type == 0){
                        value_ext -= (imgSrc(y+yi,x+xi)-128);
                    }else{
                        value_ext += (imgSrc(y+yi,x+xi)-128);
                    }
                    nbOut++;
                }
            }
        }
    }

    int total_value;

    if(nbOut == 0){
        nbOut = 1;
    }
    if(nbCenter == 0){
        total_value = 128;
    }else{
        total_value = 128 + (value_center/(float)nbCenter + value_ext/(float)nbOut)/2.0;//*cell.extern_radius;
    }


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

__global__ void directionSelectiveKernel(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, cv::cuda::PtrStepSz<u_char> imgPrev){
    // Get our global thread ID
    int xdst = blockIdx.x*blockDim.x +threadIdx.x;
    int ydst = blockIdx.y*blockDim.y+threadIdx.y;

    int nbcols = imgSrc.cols;
    int nbrows = imgSrc.rows;

    int type = xdst%4;

    int response = 0;
    int delta = 0;
    if(xdst%2 == 0){
        delta += 1;
    }else{
        delta -= 1;
    }
    int dx = xdst + delta;
    int dy = ydst + delta;

    if(type < 2 && dx < nbcols && dx >= 0){
        response = ((int)imgPrev(ydst,dx)+(int)imgSrc(ydst,xdst)) - ((int)imgSrc(ydst,dx)+(int)imgPrev(ydst,xdst));// on directive
        //response += (imgSrc(ydst,dx)+imgPrev(ydst,xdst));
    }else if(dy < nbrows && dy >= 0){
        response = ((int)imgPrev(dy,xdst)+(int)imgSrc(ydst,xdst)) - ((int)imgSrc(dy,xdst)+(int)imgPrev(ydst,xdst));// on directive
    }

    if(response<0){
        response = 0;
    }else if(response>255){
        response= 255;
    }

    //if(xdst<imgDst.cols && ydst < imgDst.rows)
        imgDst(ydst,xdst)= response;//*6;


}


__global__ void sparseKernel(cv::cuda::PtrStepSz<u_char> imgSrc, int depth, char * imgDst, unsigned char min, unsigned char max){
    // Get our global thread ID
    int xdst = blockIdx.x*blockDim.x +threadIdx.x;
    int ydst = blockIdx.y*blockDim.y+threadIdx.y;

    unsigned char sliceSize = 255/depth;
    unsigned char val = imgSrc(ydst,xdst);

    if(val>max){
        val=max;
    }else if(val<min){
        val = min;
    }else{
        val = (val-min)/(float)(max-min)*255;
    }

    unsigned char nbBytes = depth/8;


    //char amplitude = val/((float)sliceSize);
    //char extra = val % sliceSize;

    unsigned char* firstSparseData = (unsigned char*)&(imgDst[(xdst+ydst*imgSrc.cols)*depth/8]);
    int current_byte = nbBytes-1;
    unsigned char* currentSparseData = &(firstSparseData[current_byte]);

    unsigned char tempData = 0;
    for(unsigned int i=0; i<depth ;i++ ){
        if(i != 0 && i%8 == 0){
            *currentSparseData = tempData;
            current_byte--;
            currentSparseData = &(firstSparseData[current_byte]);
            tempData = 0;
        }


        if(val>sliceSize*i){
            tempData |= 1<<i%8; //Set to true
        }else{
            //tempData &= !(1<<i%8); //Set to false
        }
    }
    unsigned char dd =0;


    *currentSparseData = (char)tempData;
}

__global__ void discretiseKernel(cv::cuda::PtrStepSz<u_char> imgSrc, int depth, cv::cuda::PtrStepSz<u_char> imgDst, unsigned char min, unsigned char max){
    // Get our global thread ID
    int xdst = blockIdx.x*blockDim.x +threadIdx.x;
    int ydst = blockIdx.y*blockDim.y+threadIdx.y;

    unsigned char sliceSize = 255/depth;
    unsigned char val = imgSrc(ydst,xdst);

    if(val>max){
        val=max;
    }else if(val<min){
        val = min;
    }else{
        val = (val-min)/(float)(max-min)*255;
    }

    unsigned char newVal;
    if(val == 0){
        newVal = 0;
    }else if(val > 0 && val < 255-sliceSize){
       newVal =  (val/sliceSize+1)*sliceSize;
    }else{
       newVal = 255;
    }
    imgDst(ydst,xdst) = newVal;

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

void photoreceptorSampling1C(cv::cuda::PtrStepSz<uchar> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Cone* coneArrayGPU, int conesWidth, int conesHeight ,cudaStream_t stream)
{
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

    photoreceptorSamplingKernel1C<<<grid, block>>>(imgSrc, imgDst, coneArrayGPU, conesWidth,conesHeight);

}

void photoreceptorSampling(cv::cuda::PtrStepSz<uchar3> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Cone* coneArrayGPU, int conesWidth, int conesHeight ,cudaStream_t stream)
{
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

    //photoreceptorSamplingKernel<<<grid, block>>>(imgSrc, imgDst, coneArrayGPU, conesWidth,conesHeight);

}

void multiConvolve(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, Ganglionar* cellsArrayGPU, int cellsArrayWidth, int cellsArrayHeight,cudaStream_t stream){


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

void directionSelectiveComputation(cv::cuda::PtrStepSz<u_char> imgSrc, cv::cuda::PtrStepSz<u_char> imgDst, cv::cuda::PtrStepSz<u_char> imgPrev,cudaStream_t stream){


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

    directionSelectiveKernel<<<grid, block>>>(imgSrc, imgDst, imgPrev);

}

void sparse(cv::cuda::PtrStepSz<u_char> imgSrc, int depth, GpuBitArray2D& imgDst, unsigned char minval, unsigned char maxVal, cudaStream_t stream){
    dim3 grid, block;
    imgDst.resize(imgSrc.cols*depth,imgSrc.rows);

    // Number of threads in each thread block
    block.x = 16;
    block.y = 16;

    grid.x = (int)ceil((float)(imgSrc.cols)/block.x);
    grid.y = (int)ceil((float)(imgSrc.rows)/block.y);

    sparseKernel<<<grid, block>>>(imgSrc, depth, imgDst.data(),minval,maxVal);
}

void discretise(cv::cuda::PtrStepSz<u_char> imgSrc, int depth, cv::cuda::PtrStepSz<u_char>  imgDst, unsigned char minval, unsigned char maxVal, cudaStream_t stream){
    dim3 grid, block;

    // Number of threads in each thread block
    block.x = 16;
    block.y = 16;

    grid.x = (int)ceil((float)(imgSrc.cols)/block.x);
    grid.y = (int)ceil((float)(imgSrc.rows)/block.y);

    discretiseKernel<<<grid, block>>>(imgSrc, depth, imgDst,minval,maxVal);
}

}
