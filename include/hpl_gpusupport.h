/*
 * -- High Performance Computing Linpack Benchmark for CUDA
 *    hpl-cuda - 0.001 - 2010
 *    hpl-cuda - 0.002 - 2013
 *
 *    David Martin (cuda@avidday.net)
 *    (C) Copyright 2010-2011 All Rights Reserved
 *
 *    portions of this code written by
 *
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
 * software must display the following acknowledgement:                 
 * This  product  includes  software  developed  at  the  University  of
 * Tennessee, Knoxville, Innovative Computing Laboratory.             
 *                                                                      
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.                                                          
 *                                                                      
 * -- Disclaimer:                                                       
 *                                                                      
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 * ---------------------------------------------------------------------
 */ 
#include <cublas_v2.h>

#ifndef __GPU_SUPPORT_H__
#define __GPU_SUPPORT_H__

#define gpuQ(ans) { gpu_assert((ans), __FILE__, __LINE__); }
#define cublasQ(ans) { cublas_assert((ans), __FILE__, __LINE__); }

#define gpuPass (0)
#define gpuFail (1)
#define gpuWarn (-1)

struct gpuArray
{
    char* ptr;
    unsigned int lda;
    size_t size;
};

enum gpuUpdateStrategy
{
    gpuNone = 0,
    gpuDgemm = 1,
    gpuBoth = 2
};

struct gpuDgemmStage
{
    int N1;
    int N2;
};

struct gpuUpdatePlan
{
    enum gpuUpdateStrategy strategy;

    unsigned int           availMemory;
    struct gpuArray        gL2;
    struct gpuArray        gA;
    struct gpuArray        gL1;
    struct gpuArray        gA2;

    int                    nntot;
    int                    nnmax;
    int                    nDgemmStages;
    struct gpuDgemmStage * dgemmStages;     
};

void gpu_assert(cudaError_t code, char *file, int line);
void cublas_assert(cublasStatus_t code, char *file, int line);

cublasHandle_t cublas_handle();
void gpu_malloc_reset( );
void gpu_release( );
int gpu_init( char warmup );
int gpu_ready(unsigned int *memory );
int gpu_malloc2D( int m, int n, size_t tsize, struct gpuArray *p );
void gpu_upload( const int m, const int n, const size_t tsize, struct gpuArray *dst, const void *src, const int srclda);
void gpu_download( const int m, const int n, const size_t tsize, const void *dst, const int dstlda, struct gpuArray *src);
void gpu_copy( const int m, const int n, const size_t tsize, struct gpuArray *dst, struct gpuArray *src );

struct gpuUpdatePlan * gpuUpdatePlanCreate(int mp, int nn, int jb);
void gpuUpdatePlanDestroy(struct gpuUpdatePlan * plan);
void gpuDgemmPlanStage(struct gpuUpdatePlan * plan, const int stage, int *N1, int *N2);

#endif
