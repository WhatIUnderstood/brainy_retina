#include <cuda.h>
#include <builtin_types.h>
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>

#include "device_functions.h"
#include "cuda_arrays.h"
#include "Utils/logger.h"

#include <unistd.h>
#include <cstring>


///GPU

GpuBitArray::GpuBitArray(){
    bit_size = 0;
    byte_size = 0;
    dataOwned = true;
}

GpuBitArray::GpuBitArray(size_t size){
    bit_size = 0;
    byte_size = 0;
    dataOwned = true;
    resize(size);
}

GpuBitArray::GpuBitArray(char *data, size_t byte_size)
{
    d_buffer = data;
    this->byte_size = byte_size;
    dataOwned = false;
}

GpuBitArray::~GpuBitArray(){
    if(dataOwned && byte_size > 0){
        cudaFree(d_buffer);
    }
}

void GpuBitArray::fill(bool value){

    unsigned char default_char = 0;
    if(value){
        default_char = 255;
    }
    cudaMemset(d_buffer,default_char,byte_size);
}

size_t GpuBitArray::size(){
    return bit_size;
}

size_t GpuBitArray::bytes_size(){
    return byte_size;
}

bool GpuBitArray::resize(size_t size){
    size_t new_bit_size = size;
    size_t new_byte_size = new_bit_size/8;

    bool need_extra_byte = new_bit_size%8 != 0 ? true : false;
    if(need_extra_byte)
        new_byte_size++;

    if(bit_size != new_bit_size){
        if(byte_size > 0){
            cudaFree(d_buffer);
            byte_size = 0;
            bit_size = 0;
        }
        if(cudaSuccess != cudaMalloc(&d_buffer,new_byte_size)){
            std::cerr << "GpuBitArray::resize Error calling cudaMallocManaged" <<std::endl;
            byte_size = 0;
            bit_size = 0;
            return false;
        }else{
            byte_size = new_byte_size;
            bit_size = new_bit_size;
        }
        dataOwned = true;
    }
    return true;
}


bool GpuBitArray::dowload(HostBitArray& array, cudaStream_t stream){

    //Resize if needed
    assert(resize(array.size()));

    // copy data to CUDA device
    //cudaError_t error = cudaMemcpyAsync(d_buffer, array.data(), array.bytes_size(),cudaMemcpyHostToDevice,stream);
    cudaError_t error = cudaMemcpy(d_buffer, array.data(), array.bytes_size(),cudaMemcpyHostToDevice);
    if(cudaSuccess != error){
        logerr<<"GpuBitArray::dowload cudaMemcpyAsync size"<<array.bytes_size()<<" Error "<<std::to_string(error);
        return false;
    }

    return true;
}

bool GpuBitArray::upload(HostBitArray& array, cudaStream_t stream){

    //array.resize(this->size());
    array.resize(this->size());

    cudaError_t error = cudaMemcpy(array.data(), d_buffer, array.bytes_size(),cudaMemcpyDeviceToHost);
    if(cudaSuccess != error){
        logerr<<"GpuBitArray::upload cudaMemcpyAsync size"<<array.bytes_size()<<" Error "<<std::to_string(error);
        return false;
    }

    return true;
}

char *  GpuBitArray::data(){
    return d_buffer;
}


HostBitArray::HostBitArray(){
    byte_size = 0;
    bit_size = 0;
    buffer = 0;
}

HostBitArray::HostBitArray(size_t size, bool val):HostBitArray(){
    resize(size);
    fill(val);
}


HostBitArray::~HostBitArray()
{
    if(byte_size > 0){
        cudaFreeHost(buffer);
    }
}


void HostBitArray::fill(bool value)
{
    unsigned char default_char = 0;
    if(value){
        default_char = 255;
    }
    memset(buffer,default_char,byte_size);
}

size_t HostBitArray::size()
{
   return bit_size;
}

size_t HostBitArray::bytes_size(){
    return byte_size;
}

bool HostBitArray::resize(size_t size){
    size_t new_bit_size = size;
    size_t new_byte_size = new_bit_size/8;
    bool need_extra_byte = new_bit_size%8 != 0 ? true : false;
    if(need_extra_byte)
        new_byte_size++;

    if(new_bit_size != bit_size){
        if(bit_size > 0){
            cudaFreeHost(buffer);
            bit_size = 0;
            byte_size = 0;
        }
        cudaError_t cerror = cudaMallocHost(&buffer,new_byte_size);
        if(cudaSuccess != cerror){
            byte_size = 0;
            bit_size = 0;
            //throw CudaException("Cannot resize host array, cuda malloc failed "+std::to_string(cerror));
            return false;
        }else{
            byte_size = new_byte_size;
            bit_size = new_bit_size;
        }
    }
    return true;
}

char HostBitArray::at(size_t i){
    return buffer[i/8 + i%8] & 1<<i%8;
}

void HostBitArray::setValue(int index, bool val){
    if(val){
        buffer[index/8] |= 1<<index%8;
    }else{
        buffer[index/8] &= !(1<<index%8);
    }

}

char * HostBitArray::data(){
    return buffer;
}

//const char *CudaException::what() const _GLIBCXX_USE_NOEXCEPT
//{
//    return msg.c_str();
//}


