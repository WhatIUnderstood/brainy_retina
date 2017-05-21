#ifndef ARRAYS_H
#define ARRAYS_H

#include <vector>

template <typename T>
class BaseArray{
public:
    BaseArray(){}
    BaseArray(int size, T val){
        buffer = std::vector<T>(size,val);
    }
    ~BaseArray(){}
    void fill(T value){
        std::fill(buffer.begin(), buffer.end(), value);
    }

    unsigned long size(){return buffer.size();}
    void resize(int size){buffer.resize(size);}
    const T& at(int i){buffer.at(i);}
    void setValue(int index, T val){buffer.at(index) = val;}
private:
    std::vector<T> buffer;
};

typedef BaseArray<unsigned char> ByteArray;
//typedef BaseArray<bool> BitArray;

class BitArray{
public:
    BitArray(){}
    BitArray(int size, bool val){
        resize(size);
        fill(val);
    }
    ~BitArray(){}
    void fill(bool value){
        char default_char = 0;
        if(value){
            default_char = 255;
        }
        std::fill(buffer.begin(), buffer.end(), default_char);
    }

    unsigned long size(){return bit_size;}
    void resize(int size){
        bit_size = size;
        byte_capacity = bit_size/8;
        bool need_extra_byte = bit_size%8 != 0 ? true : false;
        if(need_extra_byte)
            byte_capacity++;
        buffer.resize(byte_capacity);
    }
    const bool& at(int i){buffer.at(i/8 + i%8) & 1<<i%8;}
    void setValue(int index, bool val){
        if(val){
            buffer.at(index/8) |= 1<<index%8;
        }else{
            buffer.at(index/8) &= !(1<<index%8);
        }
    }
private:
    std::vector<char> buffer;
    int bit_size;
    int byte_capacity;
};

#endif // ARRAYS_H
