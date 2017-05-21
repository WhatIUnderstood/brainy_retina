#ifndef CUDA_ARRAYS_H
#define CUDA_ARRAYS_H

//#include <cuda_runtime.h>
//#include "declaration_helper.cuh"
//#include "host_arrays.h"
//#include "vector_types.h"

//#include "Cuda/cuda_image.h"

//namespace cretina{

////template< typename T>
//class GpuArray{
//public:
//    GpuArray();
//    GpuArray(char * data, size_t byte_size);
//    GpuArray(size_t byte_size);
//    ~GpuArray();
//    void fill(unsigned char value);
//    size_t size();
//    bool resize(size_t size);
//    bool dowload(HostArray &array, cudaStream_t stream);
//    bool upload(HostArray &array, cudaStream_t stream);
//    char * data() const;
//    char& at(size_t i);

//private:
//    char * d_buffer;
//    //size_t bit_size;
//    size_t byte_size;
//    bool dataOwned;
//};

//class GpuImage: public GpuArray{
//public:

//    GpuImage(){
//        this->_width = 0;
//        this->_height = 0;
//        this->_depth = 0;
//    }

//    GpuImage(int width, int height, int depth, char *data):GpuArray(data,width*height*depth)
//    {
//        this->_width = width;
//        this->_height = height;
//        this->_depth = depth;
//    }

//    bool create(int width, int height, int depth){
//        this->_width = width;
//        this->_height = height;
//        this->_depth = depth;
//        GpuArray::resize(width*height*depth);
//    }

//    bool dowload(HostImage &array, cudaStream_t stream = cudaStreamDefault){
//        _width = array.width();
//        _height = array.height();
//        _depth = array.depth();
//        return GpuArray::dowload(array,stream);
//    }

//    bool upload(HostImage &array, cudaStream_t stream = cudaStreamDefault){
//        array.resize(_width,_height,_depth);
//        return GpuArray::upload(array,stream);
//    }

//    template <typename _Tp> operator PtrStepSz<_Tp>() const;
//    template <typename _Tp> operator PtrStep<_Tp>() const;




//    int width(){return _width;}
//    int height(){return _height;}
//    int depth(){return _depth;}
//private:
//    int _width;
//    int _height;
//    int _depth;

//};

//template <class T> inline
//GpuImage::operator PtrStepSz<T>() const
//{
//    return PtrStepSz<T>(_height, _width, (T*)data(), _width*_depth);
//}

//template <class T> inline
//GpuImage::operator PtrStep<T>() const
//{
//    return PtrStep<T>((T*)data(),  _width*_depth);
//}

//}



#endif // CUDA_ARRAYS_H



