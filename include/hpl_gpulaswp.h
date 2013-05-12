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


#ifndef __GPU_LASWP_H__
#define __GPU_LASWP_H__

#include "hpl_gpusupport.h"

void gpu_slaswp00N( struct gpuArray *A, const int M, const int N, const int *IPIV, const int NB, cudaStream_t stream);
void gpu_dlaswp00N( struct gpuArray *A, const int M, const int N, const int *IPIV, const int NB, cudaStream_t stream);

#endif
