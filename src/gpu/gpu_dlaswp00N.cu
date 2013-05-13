/*
 *  -- High Performance Computing Linpack Benchmark for CUDA
 *    hpl-cuda - 0.001 - 2010
 *    hpl-cuda - 0.002 - 2013
 *
 *    David Martin (cuda@avidday.net)
 *    (C) Copyright 2010-2013 All Rights Reserved
 *
 *    portions of this code written by

 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    portions of this code written by Vasily Volkov.
 *    Copyright (c) 2009, The Regents of the University of California.
 *    All rights reserved.
 *
 * -- Copyright notice and Licensing terms:
 *
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:
 *
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the 
 * documentation and/or other materials provided with the distribution.
 *
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgements:
 * This  product  includes  software  developed  at  the  University  of  
 * Tennessee, Knoxville, Innovative Computing Laboratory.
 * This product  includes software  developed at the Frankfurt Institute
 * for Advanced Studies.
 * This product includes software   developed at   the   University   of 
 * California.
 */

extern "C" {
#include "hpl_gpusupport.h"
}

#include <cstdlib>
#include <assert.h>

template<typename T>
__device__ __host__
void exchange(T &x, T &y)
{
    register T r = x; 
    x = y; 
    y = r;
}

template<typename T>
__device__ __host__
T maximum(const T &x, const T &y)
{
    return (x>y) ? x : y;
}

/*
 * Row exchange kernel for column major order storage
 */
template<typename Real>
__global__
void kernel_laswp00N( Real *A, const int M, const int N, const int LDA, 
                    const int2 *IPIV, const int NPIV)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(; row < NPIV; row += stride) {
        int2 pos = IPIV[row];
        volatile Real *a0 = A + pos.x;
        volatile Real *a1 = A + pos.y;
#pragma unroll 16
        for(int col=0; col<N; col++, a0 += LDA, a1 += LDA) {
            exchange(*a0, *a1);
        }
    }
}
        
struct laswp00N_plan
{
    int npasses;
    int *pos;
    int2 *pivot;
};

__host__
void create_laswp00n_plan(struct laswp00N_plan &plan, const int M, const int NB, const int *IPIV)
{
    int *hist = (int *)calloc(M, sizeof(int));
    int *passes = (int *)malloc(sizeof(int) * NB);
    int npasses = 0;
    for(int i=0; i<NB; i++) {
        passes[i] = maximum(hist[IPIV[i]]++, hist[i]++);
        npasses = maximum(1+passes[i], npasses);
    }
    free(hist);

    plan.npasses = npasses;
    plan.pivot = (int2 *)malloc(sizeof(int2) * NB);
    plan.pos = (int *)malloc(sizeof(int) * npasses);
    for(int pass=0, p=0; pass<npasses; pass++) {
        for(int j=0; j<NB; j++) {
            if ( (passes[j] == pass) && (j != IPIV[j]) ) {
                int2 v;
                v.x = j;
                v.y = IPIV[j];
                plan.pivot[p++] = v;
            }
        }
        plan.pos[pass] = p;
    }
    free(passes);
}

__host__
void destroy_laswp00N_plan(struct laswp00N_plan &plan)
{
    free(plan.pivot);
    free(plan.pos);
    plan.npasses = -1;
}

template<typename Real>
__host__
void driver_laswp00N( struct gpuArray *A, const int M, const int N,
                    const int *IPIV, const int NB, cudaStream_t stream)
{
    if ( (M <= 0) || (N <= 0) || (NB <= 0) ) return;

    struct laswp00N_plan plan;

    create_laswp00n_plan(plan, M, NB, IPIV );

    gpuArray _pivot;
    gpu_malloc2D(NB, 1, sizeof(int2), &_pivot);
    gpu_upload(NB, 1, sizeof(int2), &_pivot, &plan.pivot[0], NB);

    const int nthreads = 128, blocksmax = 8 * 16;
    dim3 blocksize = dim3(nthreads);

    Real *_A = (Real *)A->ptr;
    int lda = A->lda;
    for(int i=0, p=0; i<plan.npasses; i++) {
        int2 *_ipivot = (int2 *)_pivot.ptr + p;
        int n = plan.pos[i] - p;
        dim3 gridsize = max((n/nthreads) + ((n%nthreads) > 0) ? 1 : 0, blocksmax);
        kernel_laswp00N<Real><<<gridsize, blocksize, size_t(0), stream>>>(_A, M, N, lda, _ipivot, n);
        gpuQ( cudaPeekAtLastError() );
        p = plan.pos[i];
    }

    destroy_laswp00N_plan(plan);

}

extern "C"
__host__
void gpu_slaswp00N( struct gpuArray *A, const int M, const int N,
                    const int *IPIV, const int NB, cudaStream_t stream)
{
    return driver_laswp00N<float>(A, M, N, IPIV, NB, stream);

}
#if __CUDA_ARCH__ >= 130 
extern "C"
__host__
void gpu_dlaswp00N( struct gpuArray *A, const int M, const int N,
                    const int *IPIV, const int NB, cudaStream_t stream)
{
    return driver_laswp00N<double>(A, M, N, IPIV, NB, stream);

}
#endif


