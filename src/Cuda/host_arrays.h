#ifndef HOSTBITARRAY_H
#define HOSTBITARRAY_H

#include <unistd.h>
#include <exception>
#include <string>
namespace cretina{

class CudaException: public std::exception{
public:
    CudaException(){}
    CudaException(std::string msg):msg(msg){}
    const char *what() const _GLIBCXX_USE_NOEXCEPT;
    std::string msg;
};

class HostArray{
public:
    HostArray();
    HostArray(size_t size, bool val = false);
    HostArray(char* data, size_t size);
    ~HostArray();
    void fill(unsigned char value);
    size_t size();
    bool resize(size_t size);
    char& at(size_t i);

    void setValue(int index, char val);
    char * data();
    //bool download(char* array, cudaStream_t stream);
private:
    char * buffer;
    size_t byte_size;
    bool dataOwned;
};


class HostImage: public HostArray{
public:
    HostImage(){
        this->_width = 0;
        this->_height = 0;
        this->_depth = 0;
    }

    HostImage(int width, int height, int depth, char *data):HostArray(data,width*height*depth)
    {
        this->_width = width;
        this->_height = height;
        this->_depth = depth;
    }
    HostImage(int width, int height, int depth)
    {
        resize(width,height,depth);
    }
    void resize(int width, int height, int depth);
    void copyFrom(int width, int height, int depth, char *data);

    int width(){return _width;}
    int height(){return _height;}
    int depth(){return _depth;}
private:
    int _width;
    int _height;
    int _depth;

};

}

#endif // HOSTBITARRAY_H
