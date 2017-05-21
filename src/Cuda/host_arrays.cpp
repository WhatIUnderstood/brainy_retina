#include "host_arrays.h"

#include <unistd.h>
#include <cstring>
#include <cuda_runtime.h>

#include <exception>

namespace cretina {

HostArray::HostArray(){
    byte_size = 0;
    buffer = 0;
    dataOwned = true;
}

HostArray::HostArray(size_t size, bool val)
{
    resize(size);
    fill(val);
    dataOwned = true;
}

HostArray::HostArray(char *data, size_t size):HostArray(){
    dataOwned = false;
}

HostArray::~HostArray()
{
    if(dataOwned && byte_size > 0){
        cudaFreeHost(buffer);
    }
}


void HostArray::fill(unsigned char value)
{
    memset(buffer,value,byte_size);
}


size_t HostArray::size(){
    return byte_size;
}

bool HostArray::resize(size_t size){
    size_t new_byte_size = size;

    if(new_byte_size != byte_size){
        if(dataOwned && byte_size > 0){
            cudaFreeHost(buffer);
            byte_size = 0;
        }
        dataOwned = true;
        cudaError_t cerror = cudaMallocHost(&buffer,new_byte_size);
        if(cudaSuccess != cerror){
            byte_size = 0;
            throw CudaException("Cannot resize host array, cuda malloc failed "+std::to_string(cerror));
            //return false;
        }else{
            byte_size = new_byte_size;
        }
    }
    return true;
}

char &HostArray::at(size_t i)
{
    return buffer[i];
}



void HostArray::setValue(int index, char val){
    buffer[index] = val;
}

char * HostArray::data(){
    return buffer;
}

const char *CudaException::what() const _GLIBCXX_USE_NOEXCEPT
{
    return msg.c_str();
}

void HostImage::resize(int width, int height, int depth)
{
    this->_width = width;
    this->_height = height;
    this->_depth = depth;
    HostArray::resize(width*height*depth);
}

void HostImage::copyFrom(int width, int height, int depth, char *_data)
{
    resize(width,height,depth);
    memcpy(data(),_data,width*height*depth);
}



}

