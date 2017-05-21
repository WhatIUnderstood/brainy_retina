//#include <cuda.h>
//#include <builtin_types.h>
//#include <math.h>
//#include <cuda_runtime.h>
//#include <iostream>
//#include <assert.h>

//#include "device_functions.h"
//#include "cuda_arrays.h"
//#include "Utils/logger.h"

//#include <unistd.h>
//#include <cstring>

//namespace cretina {

/////GPU

//GpuArray::GpuArray(){
//    byte_size = 0;
//    dataOwned = true;
//}

//GpuArray::GpuArray(size_t size){
//    byte_size = 0;
//    dataOwned = true;
//    resize(size);
//}

//GpuArray::GpuArray(char *data, size_t byte_size)
//{
//    d_buffer = data;
//    this->byte_size = byte_size;
//    dataOwned = false;
//}

//GpuArray::~GpuArray(){
//    if(dataOwned && byte_size > 0){
//        cudaFree(d_buffer);
//    }
//}

//void GpuArray::fill(unsigned char value){

//    cudaMemset(d_buffer,value,byte_size);
//}


//size_t GpuArray::size(){
//    return byte_size;
//}

//bool GpuArray::resize(size_t size){
//    size_t new_byte_size = size;

//    if(byte_size != new_byte_size){
//        if(byte_size > 0){
//            cudaFree(d_buffer);
//            byte_size = 0;
//        }
//        if(cudaSuccess != cudaMalloc(&d_buffer,new_byte_size)){
//            std::cerr << "GpuArray::resize Error calling cudaMallocManaged" <<std::endl;
//            byte_size = 0;
//            return false;
//        }else{
//            byte_size = new_byte_size;
//        }
//        dataOwned = true;
//    }
//    return true;
//}


//bool GpuArray::dowload(HostArray& array, cudaStream_t stream){

//    //Resize if needed
//    assert(resize(array.size()));

//    // copy data to CUDA device
//    //cudaError_t error = cudaMemcpyAsync(d_buffer, array.data(), array.bytes_size(),cudaMemcpyHostToDevice,stream);
//    cudaError_t error = cudaMemcpy(d_buffer, array.data(), array.size(),cudaMemcpyHostToDevice);
//    if(cudaSuccess != error){
//        logerr<<"GpuArray::dowload cudaMemcpyAsync size"<<array.size()<<" Error "<<std::to_string(error);
//        return false;
//    }

//    return true;
//}

//bool GpuArray::upload(HostArray& array, cudaStream_t stream){

//    //Resize if needed
//    assert(array.resize(size()));

//    cudaError_t error = cudaMemcpy(array.data(),d_buffer , size(),cudaMemcpyDeviceToHost);
//    if(cudaSuccess != error){
//        logerr<<"GpuArray::upload cudaMemcpyAsync size"<<array.size()<<" Error "<<std::to_string(error);
//        return false;
//    }

//    return true;
//}

//char *  GpuArray::data() const{
//    return d_buffer;
//}

//char &GpuArray::at(size_t i)
//{
//    return d_buffer[i];
//}

//}




