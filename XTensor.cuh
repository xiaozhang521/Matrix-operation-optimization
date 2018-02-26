/* NiuTrans - an open-source MT toolkit
 * Copyright (C) 2017, Natural Language Processing Lab. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
 */

/*
 * $Id:
 * XTensor; XTensor.h
 * implementation of tensors used in this work. It it is the basis of XMatrix and
 * XVector
 *
 * $Version:
 * 0.1.0
 *
 * $Created by:
 * XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2017-07-31
 * I'm working while most of the students are enjoying their holidays :(
 * $Update by:
 * LI Yinqiao (email: 1023907632@qq.com) 2017-11-18 bug fixes
 *
 */

#ifndef __XTENSOR_CUH__
#define __XTENSOR_CUH__

#ifdef USE_CUDA

#include "XTensor.h"

/**************************************/
/* copy all elements from a source matrix to a target matrix */
extern "C"
bool CudaCopyValues(XTensor * s, XTensor * t, XStream * stream = NULL);

/**************************************/
/* flush a list of XTensor to GPU memory */
void CudaCPUToGPUFlush(List * mList, XMem * GPUMem);

/* copy the data from GPU memory to CPU memory */
void CudaGPUToCPUFlush(XTensor * tensor);


/**************************************/
/* set the cell to the ascending order along a given dimension */
extern "C"
void CudaSetAscendingOrder(XTensor * a, int dim);

/**************************************/
/* set each entry to its negtive value (CUDA Kernel) */
__global__ 
void KernelNegate(DTYPE * d, int size);

/* set each entry to its negtive value (CUDA Kernel) with float16 data type*/
__global__ 
void KernelNegate(__half * d, int size);

/* set each entry to its negtive value */
extern "C"
void CudaNegate(XTensor * a);

/**************************************/
/* set all entries to its root (CUDA Kernel) */
__global__
void KernelSqrtV2(DTYPE * d, int size);

/* set all entries to its root (CUDA Kernel) */
__global__
void KernelSqrtV2(__half * d, int size);

/* get the power of the entries */
extern "C"
void CudaPower(XTensor * a, DTYPE p);

/**************************************/
/* scale and shift all matrix entires p = p * scale + shift (CUDA Kernel) */
__global__ 
void KernelScaleAndShift(DTYPE * d, int size, DTYPE scale, DTYPE shift);

/* scale and shift all matrix entires p = p * scale + shift (CUDA Kernel) with float16 data type */
__global__ 
void KernelScaleAndShift(__half * d, int size, __half scale, __half shift);

/* scale and shift all tensor entires */
extern "C" 
void CudaScaleAndShift(XTensor * a, DTYPE scale, DTYPE shift);

/**************************************/
/* copy a number of blocks to target positions */
__global__ 
void KernelCopyBlocks(DTYPE * source, int blockSize, int blockNum, DTYPE * target, int * targetBlocks);

/* copy a number of blocks to target positions (cuda version) */
extern "C" 
void CudaCopyBlocks(void * source, int blockSize, int blockNum, void * target, int * targetBlocks, XMem * myMem);

/**************************************/
/* copy a number of blocks form source positions to target positions */
__global__ 
void KernelCopyBlocksSelected(DTYPE * source, int blockSize, int * sourceBlocks, int blockNum, DTYPE * target, int * targetBlocks);

/* copy a number of blocks form source positions to target positions (cuda version) */
extern "C" 
void CudaCopyBlocksSelected(void * source, int blockSize, int * sourceBlocks, int blockNum, void * target, int * targetBlocks, XMem * myMem);

/**************************************/
/* set target data block index for the data movement in split */
extern "C" 
void CudaMakeSplitBlockIndex(int devID, int * blockIndex, int splitNum, int blockSplitSize, int blockNum);

/**************************************/

/* copy a number of blocks (of different sizes) to target positions */
__global__ 
void KernelCopyBlockLists(DTYPE ** sourceList, int * sourceBlockSizes, int sourceBlockNum, DTYPE ** targetList);

/* merge data by blocks (cuda version) */
extern "C" 
void CudaMergeBlockLists(List * sourceList, int * blockSizes, int blockNum, void * target, XMem * myMem);

/**************************************/
/* set target data block index for the data movement in split */
extern "C" 
void CudaMakeMergeBlockIndex(int devID, 
                             int * blockIndex, int blockNum, int blockNumInMerge, 
                             int splitSizeInGrid, int gridSize, int gridNum);

/**************************************/
/* duplicate the data along a given dimension */
extern "C"
void CudaUnsqueeze(XTensor * a, XTensor * b, int dim, int dSize);

/**************************************/
/* sort the tensor along a given dimension */
void CudaSortBig(XTensor * a, XTensor * b, XTensor * indexA, XTensor * indexB, int dim, int k = -1);

/**************************************/
/* get the top-k items along a given dimension */
void CudaTopK(XTensor * a, XTensor * b, XTensor * index, int dim, int k);
/**************************************/
/**************************************/
/* get the top-k items along a given dimension, use radixSelect algorithm */
void CudaTopKRadixSelect(XTensor * a, XTensor * b, XTensor * index, int dim, int k);
/**************************************/

/* summation of data arrays (CUDA Kernel) */
extern "C" __global__ 
void KernelADD(DTYPE * a, DTYPE * b, DTYPE * c, int size, DTYPE beta = (DTYPE)1.0);

/* tensor summation c = a + b * \beta (cuda version) */
extern "C" 
void CudaSum(XTensor * a, XTensor * b, XTensor * c = NULL, DTYPE beta = (DTYPE)1.0);

/*  tensor summation c = a + b * \beta (cuda version) with an input handle */
extern "C" 
void CudaSumWithHandle(int devID, cublasHandle_t * handle, DTYPE * a, DTYPE * b, DTYPE * c, int size, DTYPE beta = (DTYPE)1.0);

/**************************************/
/* summation of a tensor and a vector (column vector) */
extern "C"
void CudaSumByColumnTV(XTensor * a, XTensor * b, XTensor * c, DTYPE beta = (DTYPE)1.0);

/**************************************/
/* summation of a vector (column vector) and a tensor */
extern "C"
void CudaSumByColumnVT(XTensor * a, XTensor * b, XTensor * c, DTYPE beta = (DTYPE)1.0);


/**************************************/
/* 
mutilication of a dense matrix with a sparse vector 
c = a * b * \alpha
*/
extern "C" __global__
void KernelMatrixMulDenseMSparseMV2(DTYPE * a, MATRIX_TRANS_TYPE transposedA, int aColSize, int aRowSize, 
                                    void * b, MATRIX_TRANS_TYPE transposedB, int bNonZeroNum,int bColSize, int bRowSize, 
                                    DTYPE * c, int cColSize, int cRowSize, DTYPE alpha);

/* 
matrix multiplication (for 2d tensors) (cuda version)
c = trans(a) * trans(b) * alpha + c * beta 
where trans() return the transposed matrix if the flag is fired
*/
extern "C" 
void CudaMatrixMul2D(XTensor * a, MATRIX_TRANS_TYPE transposedA, XTensor * b, MATRIX_TRANS_TYPE transposedB, XTensor * c, 
                        DTYPE alpha = (DTYPE)1.0, DTYPE beta = 0, XStream * stream = NULL);


/**************************************/

/* multiplication of two tensors in a element-wise manner c(i) = a(i)*b(i) */
extern "C" __global__ 
void KernelMulElementWise(DTYPE * a, DTYPE * b, DTYPE * c, int size);

/* multiplication of two tensors in a element-wise manner c(i) = a(i)*b(i) + \alpha*c(i) */
extern "C" __global__ 
void KernelMulElementWiseV2(DTYPE * a, DTYPE * b, DTYPE * c, int size, DTYPE alpha);

/* multiplication of two tensors in a element-wise manner c(i) = a(i)*b(i)+ \alpha*c(i)  */
template<int nonZeroAlpha>__global__ 
void KernelMulElementWiseTensorDynamic(DTYPE * a, DTYPE * b, DTYPE * c, DTYPE alpha, int stride, int ldSizeA, int ldSizeB, int ldSizeC, int blockNum);

/* element-wise product of two tensors */
extern "C" 
void CudaMultiplyElementWise(XTensor * a, XTensor * b, XTensor * c, int leadingDim, DTYPE alpha);

/**************************************/
/* 
sum the items along a dimension of the tensor (cuda version) 
For a 1-dimensional data array a,
sum = \sum_i ((a_i + shift)^power) if isExp == false
sum = \sum_i exp((a_i + shift)^power) if isExp == true
*/
extern "C" 
void CudaReduceSumXT(XTensor * input, XTensor * output, int dim, XTensor * shift, DTYPE power, bool isExp);

/**************************************/
/* get the max-valued items along a dimension of the tensor (cuda version) */
extern "C" 
void CudaReduceMaxXT(XTensor * input, XTensor * output, int dim);

/**************************************/
/* normalized the data with normal distribution (Kernel code). For an input x,
   y = a * (x-mean)/sqrt(variance+\epsilon) + b
   where a and b are the scalar and bias respectively, and \epsilon is the adjustment parameter
*/
__global__
void KernelNormalize(DTYPE * input, DTYPE * output, DTYPE * mean, DTYPE * var,
                     DTYPE * a, DTYPE * b, DTYPE epsilon, 
                     int stride, int strideNum, int blockNum);

/* normalized the data with normal distribution. For an input x,
   y = a * (x-mean)/sqrt(variance+\epsilon) + b
   where a and b are the scalar and bias respectively, and \epsilon is the adjustment parameter
*/
extern "C" 
void CudaNormalize(XTensor * input, XTensor * output, int dim, 
                   XTensor * mean, XTensor * var, 
                   XTensor * a, XTensor * b, DTYPE epsilon);

#endif

#endif
