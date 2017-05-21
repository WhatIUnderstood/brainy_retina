#ifndef CUDA_H_H
#define CUDA_H_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#endif // CUDA_H_H

