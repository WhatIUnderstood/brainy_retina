#ifndef CUDA_ARRAYS_H
#define CUDA_ARRAYS_H


#include "declaration_helper.cuh"

class HostBitArray{
public:
    HostBitArray();
    HostBitArray(size_t size, bool val = false);
    ~HostBitArray();
    void fill(bool value);
    size_t size();
    size_t bytes_size();
    bool resize(size_t size);
    char at(size_t i);
    void setValue(int index, bool val);
    char * data();
    //bool download(char* array, cudaStream_t stream);
private:
    char * buffer;
    size_t bit_size;
    size_t byte_size;
};


class GpuBitArray{
public:
    GpuBitArray();
    GpuBitArray(char * data, size_t byte_size);
    GpuBitArray(size_t size);
    ~GpuBitArray();
    void fill(bool value);
    size_t size();
    size_t bytes_size();
    bool resize(size_t size);
    bool dowload(HostBitArray& array);
    bool upload(HostBitArray& array);
    char * data();

private:
    char * d_buffer;
    size_t bit_size;
    size_t byte_size;
    bool dataOwned;
};

class HostBitArray2D: public HostBitArray{
public:
    HostBitArray2D():HostBitArray(){bit_width = 0; bit_height = 0;}
    HostBitArray2D(size_t bit_width, size_t bit_height):HostBitArray(bit_width*bit_height),bit_width(bit_width),bit_height(bit_height){}
    bool resize(size_t size){bit_width = size; bit_height = 1;return HostBitArray::resize(size);}
    bool resize(size_t w,size_t h){bit_width = w; bit_height = h;return HostBitArray::resize(w*h);}
    int width(){return bit_width;}
    int height(){return bit_height;}
    int bytesWidth(){return bit_width/8;}
    int bytesHeight(){return bit_height;}
private:
    friend class GpuBitArray2D;
    size_t bit_width;
    size_t bit_height;
};

class GpuBitArray2D: public GpuBitArray{
public:
    GpuBitArray2D():GpuBitArray(){bit_width = 0; bit_height = 0;}
    GpuBitArray2D(char * data, size_t bit_width, size_t bit_height):GpuBitArray(data,bit_width*bit_height),bit_width(bit_width),bit_height(bit_height){}
    bool resize(size_t size){bit_width = size; bit_height = 1;return GpuBitArray::resize(size);}
    bool resize(size_t w,size_t h){bit_width = w; bit_height = h;return GpuBitArray::resize(w*h);}
    bool dowload(HostBitArray& array){return GpuBitArray::dowload(array);}
    bool upload(HostBitArray2D& array){array.bit_width=bit_width;array.bit_height=bit_height;return GpuBitArray::upload(array);}
private:
    size_t bit_width;
    size_t bit_height;
};



#endif // CUDA_ARRAYS_H



