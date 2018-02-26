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
 * $Update by:
 * LI Yinqiao (email: 1023907632@qq.com) 2017-11-18 bug fixes
 *
 */

#include "XTensor.h"
#include "XTensor.cuh"
#include "XDevice.h"
#include "XHeap.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

/************************************************************
* basic kernels
*/

/* 
matrix multiplication via cuda version BLAS
*/

void CudaBLASMatrixMUL(cublasHandle_t * handle, 
                       void * a, MATRIX_TRANS_TYPE transposedA, MATRIX_DATA_TYPE dataTypeA, 
                       void * b, MATRIX_TRANS_TYPE transposedB, MATRIX_DATA_TYPE dataTypeB, 
                       void * c, MATRIX_DATA_TYPE dataTypeC, 
                       int na, int ma, int nb, int mb, int nc, int mc, 
                       DTYPE alpha, DTYPE beta)
{
    /* 
    matrxi-matrix multiplication 
    For row-major matrices (as in c/c++), the trick used here is (AB)^T = B^T * A^T
    */
    if(dataTypeA == X_DOUBLE && dataTypeB == X_DOUBLE && dataTypeC == X_DOUBLE){
        double alpha2 = (double)alpha;
        double beta2 = (double)beta;
        if(transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasDgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const double*)b, mb, (const double*)a, ma, &beta2, (double*)c, mc);
        else if(transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasDgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, &alpha2, (const double*)b, mb, (const double*)a, ma, &beta2, (double*)c, mc);
        else if(transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasDgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const double*)b, mb, (const double*)a, ma, &beta2, (double*)c, mc);
        else if(transposedA == X_TRANS && transposedB == X_TRANS)
            cublasDgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, &alpha2, (const double*)b, nb, (const double*)a, ma, &beta2, (double*)c, mc);
    }
    else if(dataTypeA == X_FLOAT && dataTypeB == X_FLOAT && dataTypeC == X_FLOAT){
        float alpha2 = (float)alpha;
        float beta2 = (float)beta;
        if(transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const float*)b, mb, (const float*)a, ma, &beta2, (float*)c, mc);
        else if(transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, &alpha2, (const float*)b, mb, (const float*)a, ma, &beta2, (float*)c, mc);
        else if(transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasSgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const float*)b, mb, (const float*)a, ma, &beta2, (float*)c, mc);
        else if(transposedA == X_TRANS && transposedB == X_TRANS)
            cublasSgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, &alpha2, (const float*)b, mb, (const float*)a, ma, &beta2, (float*)c, mc);
    }
    else if(dataTypeA == X_FLOAT16 && dataTypeB == X_FLOAT16 && dataTypeC == X_FLOAT16){
        unsigned short alpha2 = FloatToFloat16(alpha);
        unsigned short beta2 = FloatToFloat16(beta);
        __half * alpha3 = (__half*)&alpha2;
        __half * beta3 = (__half*)&beta2;
        if(transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasHgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, alpha3, (const __half*)b, mb, (const __half*)a, ma, beta3, (__half*)c, mc);
        else if(transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasHgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, alpha3, (const __half*)b, mb, (const __half*)a, ma, beta3, (__half*)c, mc);
        else if(transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasHgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, alpha3, (const __half*)b, mb, (const __half*)a, ma, beta3, (__half*)c, mc);
        else if(transposedA == X_TRANS && transposedB == X_TRANS)
            cublasHgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, alpha3, (const __half*)b, mb, (const __half*)a, ma, beta3, (__half*)c, mc);
    }
    else{
        ShowNiuTransErrors("Unsupported data type!");
    }
}

/* 
matrix multiplication via cuda version BLAS
*/
void CudaBLASMatrixMULBatched(cublasHandle_t * handle, 
                              const void ** a, MATRIX_TRANS_TYPE transposedA, MATRIX_DATA_TYPE dataTypeA, 
                              const void ** b, MATRIX_TRANS_TYPE transposedB, MATRIX_DATA_TYPE dataTypeB, 
                              void ** c, MATRIX_DATA_TYPE dataTypeC, 
                              int count, int na, int ma, int nb, int mb, int nc, int mc, 
                              DTYPE alpha, DTYPE beta)
{
    /* 
    matrxi-matrix multiplication 
    For row-major matrices (as in c/c++), the trick used here is (AB)^T = B^T * A^T
    */
    if(dataTypeA == X_DOUBLE && dataTypeB == X_DOUBLE && dataTypeC == X_DOUBLE){
        double alpha2 = (double)alpha;
        double beta2 = (double)beta;
        if(transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasDgemmBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, &alpha2,(const double**)b, mb, (const double**)a, ma, &beta2, (double**)c, mc, count);
        else if(transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasDgemmBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, &alpha2,(const double**)b, mb, (const double**)a, ma, &beta2, (double**)c, mc, count);
        else if(transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasDgemmBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, &alpha2,(const double**)b, mb, (const double**)a, ma, &beta2, (double**)c, mc, count);
        else if(transposedA == X_TRANS && transposedB == X_TRANS)
            cublasDgemmBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, &alpha2,(const double**)b, nb, (const double**)a, ma, &beta2, (double**)c, mc, count);   
    }
    else if(dataTypeA == X_FLOAT && dataTypeB == X_FLOAT && dataTypeC == X_FLOAT){
        float alpha2 = (float)alpha;
        float beta2 = (float)beta;
        if(transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasSgemmBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const float**)b, mb, (const float**)a, ma, &beta2, (float**)c, mc, count);
        else if(transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasSgemmBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, &alpha2, (const float**)b, mb, (const float**)a, ma, &beta2, (float**)c, mc, count);
        else if(transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasSgemmBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const float**)b, mb, (const float**)a, ma, &beta2, (float**)c, mc, count);
        else if(transposedA == X_TRANS && transposedB == X_TRANS)
            cublasSgemmBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, &alpha2, (const float**)b, mb, (const float**)a, ma, &beta2, (float**)c, mc, count);
    }
    else if(dataTypeA == X_FLOAT16 && dataTypeB == X_FLOAT16 && dataTypeC == X_FLOAT16){
        unsigned short alpha2 = FloatToFloat16(alpha);
        unsigned short beta2 = FloatToFloat16(beta);
        __half * alpha3 = (__half*)&alpha2;
        __half * beta3 = (__half*)&beta2;
        if(transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasHgemmBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, alpha3, (const __half**)b, mb, (const __half**)a, ma, beta3, (__half**)c, mc, count);
        else if(transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasHgemmBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, alpha3, (const __half**)b, mb, (const __half**)a, ma, beta3, (__half**)c, mc, count);
        else if(transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasHgemmBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, alpha3, (const __half**)b, mb, (const __half**)a, ma, beta3, (__half**)c, mc, count);
        else if(transposedA == X_TRANS && transposedB == X_TRANS)
            cublasHgemmBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, alpha3, (const __half**)b, mb, (const __half**)a, ma, beta3, (__half**)c, mc, count);
    }
    else{
        ShowNiuTransErrors("Unsupported data type!");
    }
}

/* matrix multiplication in batch and strided mode via cuda version BLAS */
extern "C" 
void CudaBLASMatrixMULBatchedStrided(cublasHandle_t * handle, 
                                     const void * a, MATRIX_TRANS_TYPE transposedA, MATRIX_DATA_TYPE dataTypeA, long long int strideA, 
                                     const void * b, MATRIX_TRANS_TYPE transposedB, MATRIX_DATA_TYPE dataTypeB, long long int strideB, 
                                     void * c, MATRIX_DATA_TYPE dataTypeC, long long int strideC,
                                     int count, int na, int ma, int nb, int mb, int nc, int mc, 
                                     DTYPE alpha, DTYPE beta)
{
    /* 
    matrxi-matrix multiplication 
    For row-major matrices (as in c/c++), the trick used here is (AB)^T = B^T * A^T
    */
    if(dataTypeA == X_DOUBLE && dataTypeB == X_DOUBLE && dataTypeC == X_DOUBLE){
        double alpha2 = (double)alpha;
        double beta2 = (double)beta;
        if(transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasDgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const double*)b, mb, strideB, (const double*)a, ma, strideA, &beta2, (double*)c, mc, strideC, count);
        else if(transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasDgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, &alpha2, (const double*)b, mb, strideB, (const double*)a, ma, strideA, &beta2, (double*)c, mc, strideC, count);
        else if(transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasDgemmStridedBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const double*)b, mb, strideB, (const double*)a, ma, strideA, &beta2, (double*)c, mc, strideC, count);
        else if(transposedA == X_TRANS && transposedB == X_TRANS)
            cublasDgemmStridedBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, &alpha2, (const double*)b, nb, strideB, (const double*)a, ma, strideA, &beta2, (double*)c, mc, strideC, count);
    }
    else if(dataTypeA == X_FLOAT && dataTypeB == X_FLOAT && dataTypeC == X_FLOAT){
        float alpha2 = (float)alpha;
        float beta2 = (float)beta;
        if(transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasSgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const float*)b, mb, strideB, (const float*)a, ma, strideA, &beta2, (float*)c, mc, strideC, count);
        else if(transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasSgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, &alpha2, (const float*)b, mb, strideB, (const float*)a, ma, strideA, &beta2, (float*)c, mc, strideC, count);
        else if(transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasSgemmStridedBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const float*)b, mb, strideB, (const float*)a, ma, strideA, &beta2, (float*)c, mc, strideC, count);
        else if(transposedA == X_TRANS && transposedB == X_TRANS)
            cublasSgemmStridedBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, &alpha2, (const float*)b, mb, strideB, (const float*)a, ma, strideA, &beta2, (float*)c, mc, strideC, count);
    }
    else if (dataTypeA == X_FLOAT16 && dataTypeB == X_FLOAT16 && dataTypeC == X_FLOAT16) {
        unsigned short alpha2 = FloatToFloat16(alpha);
        unsigned short beta2 = FloatToFloat16(beta);
        __half * alpha3 = (__half*)&alpha2;
        __half * beta3 = (__half*)&beta2;
        if (transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasHgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, (const __half*)alpha3, (const __half*)b, mb, strideB, (const __half*)a, ma, strideA, (const __half*)beta3, (__half*)c, mc, strideC, count);
        else if (transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasHgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, (const __half*)alpha3, (const __half*)b, mb, strideB, (const __half*)a, ma, strideA, (const __half*)beta3, (__half*)c, mc, strideC, count);
        else if (transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasHgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, (const __half*)alpha3, (const __half*)b, mb, strideB, (const __half*)a, ma, strideA, (const __half*)beta3, (__half*)c, mc, strideC, count);
        else if (transposedA == X_TRANS && transposedB == X_TRANS)
            cublasHgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, (const __half*)alpha3, (const __half*)b, mb, strideB, (const __half*)a, ma, strideA, (const __half*)beta3, (__half*)c, mc, strideC, count);
    }
    else{
        ShowNiuTransErrors("Unsupported data type!");
    }
}

/* 
matrix multiplication via cuda version BLAS
*/
void CudaBLASMatrixMULList(cublasHandle_t * handle, 
                           List * a, MATRIX_TRANS_TYPE transposedA, 
                           List * b, MATRIX_TRANS_TYPE transposedB, 
                           List * c, 
                           int count, DTYPE alpha, DTYPE beta)
{
    CheckNiuTransErrors((a && b && c), "Empty input lists!");
    CheckNiuTransErrors((a->count == b->count && a->count == c->count), "Input lists must be of the same size!");

    if(a->count == 0)
        return;

    bool isUniform = true;
    bool isStrided = true;
    MTYPEINT strideA = MAX_INT;
    MTYPEINT strideB = MAX_INT;
    MTYPEINT strideC = MAX_INT;
    for(int i = 1; i < a->count; i++){
        XTensor * aim = (XTensor*)a->GetItem(i - 1);
        XTensor * bim = (XTensor*)b->GetItem(i - 1);
        XTensor * cim = (XTensor*)c->GetItem(i - 1);
        XTensor * ai = (XTensor*)a->GetItem(i);
        XTensor * bi = (XTensor*)b->GetItem(i);
        XTensor * ci = (XTensor*)c->GetItem(i);
        if(!XTensor::IsIdentical(aim, ai) || 
           !XTensor::IsIdentical(bim, bi) ||
           !XTensor::IsIdentical(cim, ci))
        {
            isUniform = false;
            break;
        }
        if(isStrided){
            MTYPEINT gapA = MTYPEINT(ai->data) - MTYPEINT(aim->data);
            MTYPEINT gapB = MTYPEINT(bi->data) - MTYPEINT(bim->data);
            MTYPEINT gapC = MTYPEINT(ci->data) - MTYPEINT(cim->data);

            if(strideA == MAX_INT)
                strideA = gapA;
            if(strideB == MAX_INT)
                strideB = gapB;
            if(strideC == MAX_INT)
                strideC = gapC;

            if(strideA != gapA || strideB != gapB || strideC != gapC)
                isStrided = false;
        }
    }
    XTensor * a0 = (XTensor*)a->GetItem(0);
    XTensor * b0 = (XTensor*)b->GetItem(0);
    XTensor * c0 = (XTensor*)c->GetItem(0);

    if(isUniform){
        XMem * mem = a0->mem;
        if(isStrided){
            CudaBLASMatrixMULBatchedStrided(handle, 
                                            a0->data, transposedA, a0->dataType, strideA/a0->unitSize,
                                            b0->data, transposedB, b0->dataType, strideB/b0->unitSize,
                                            c0->data, c0->dataType, strideC/c0->unitSize, a->count, 
                                            a0->dimSize[1], a0->dimSize[0], 
                                            b0->dimSize[1], b0->dimSize[0],
                                            c0->dimSize[1], c0->dimSize[0], alpha, beta);
        }
        else{
            DTYPE ** ap = new DTYPE*[a->count];
            DTYPE ** bp = new DTYPE*[b->count];
            DTYPE ** cp = new DTYPE*[c->count];

            for(int i = 0; i < a->count; i++){
                XTensor * ai = (XTensor*)a->GetItem(i);
                XTensor * bi = (XTensor*)b->GetItem(i);
                XTensor * ci = (XTensor*)c->GetItem(i);
                ap[i] = (DTYPE*)ai->data;
                bp[i] = (DTYPE*)bi->data;
                cp[i] = (DTYPE*)ci->data;
            }

            mem->SetPinBuf();
            DTYPE ** apGPU = (DTYPE**)mem->AllocBuf(mem->devID, sizeof(DTYPE*) * a->count, 256);
            DTYPE ** bpGPU = (DTYPE**)mem->AllocBuf(mem->devID, sizeof(DTYPE*) * a->count, 256);
            DTYPE ** cpGPU = (DTYPE**)mem->AllocBuf(mem->devID, sizeof(DTYPE*) * a->count, 256);

            cudaMemcpy(apGPU, ap, sizeof(DTYPE*) * a->count, cudaMemcpyHostToDevice);
            cudaMemcpy(bpGPU, bp, sizeof(DTYPE*) * b->count, cudaMemcpyHostToDevice);
            cudaMemcpy(cpGPU, cp, sizeof(DTYPE*) * c->count, cudaMemcpyHostToDevice);

            CudaBLASMatrixMULBatched(handle, 
                                    (const void**)apGPU, transposedA, a0->dataType,
                                    (const void**)bpGPU, transposedB, b0->dataType,
                                    (void**)cpGPU, c0->dataType, a->count, 
                                     a0->dimSize[1], a0->dimSize[0], 
                                     b0->dimSize[1], b0->dimSize[0],
                                     c0->dimSize[1], c0->dimSize[0], alpha, beta);
            delete[] ap;
            delete[] bp;
            delete[] cp;

            mem->BackToPinBuf();
        }

    }
    else{
        for(int i = 0; i < a->count; i++){
            XTensor * ai = (XTensor*)a->GetItem(i);
            XTensor * bi = (XTensor*)b->GetItem(i);
            XTensor * ci = (XTensor*)c->GetItem(i);

            CudaBLASMatrixMUL(handle, 
                              ai->data, transposedA, ai->dataType, 
                              bi->data, transposedB, bi->dataType, 
                              ci->data, ci->dataType, 
                              ai->dimSize[1], ai->dimSize[0], 
                              bi->dimSize[1], bi->dimSize[0], 
                              ci->dimSize[1], ci->dimSize[0], alpha, beta);
        }
    }
}

__global__ 
void KernelFloatToFloat16(float * s, __half * t, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        t[i] = __float2half(s[i]);
    }
}

__global__ 
void KernelFloat16ToFloat(__half * s, float * t, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        t[i] = __half2float(s[i]);
    }
}


/* 
data conversion (cuda code) 
>> devID - device id
>> s - source data array
>> typeS - source data type
>> t - target data array
>> typeT - target data type
>> size - number of the items in s (and t)
*/
void CudaConvertDataType(int devID, void * s, MATRIX_DATA_TYPE typeS, void * t, MATRIX_DATA_TYPE typeT, int size)
{
    CheckNiuTransErrors((devID >= 0), "This code must be run on GPUs!");

    if(typeS == typeT)
        return;

    int gridSize[3];
    int blockSize[3];

    GDevs->GetGridAndBlockSize(devID, size, gridSize, blockSize);

    dim3 blocks(gridSize[0]);
    dim3 threads(blockSize[0]);

    if(typeS == X_FLOAT && typeT == X_FLOAT16)
        KernelFloatToFloat16<<<blocks, threads>>>((float*)s, (__half*)t, size);
    else if(typeS == X_FLOAT16 && typeT == X_FLOAT)
        KernelFloat16ToFloat<<<blocks, threads>>>((__half*)s, (float*)t, size);
    else{
        ShowNiuTransErrors("Unsupported data types for conversion!");
    }
}

/************************************************************
* cuda means
*/

/* 
copy a range of elements from a source vector to a target vector 
>> s - source matrix
>> t - target matrix
>> stream - the stream for creating the job pipeline
<< return - succeed or not
*/
bool CudaCopyValues(XTensor * s, XTensor * t, XStream * stream)
{
    if(s == NULL || t == NULL)
        return false;

    CheckNiuTransErrors(s->dataType == t->dataType, "Unmatched data type!");
    CheckNiuTransErrors((s->unitSize == t->unitSize), "Incompatible vectors in value copy.");
    CheckNiuTransErrors((s->denseRatio <= s->denseRatio), "Incompatible vectors in value copy.");

    /* dense -> dense */
    if(!s->isSparse && !t->isSparse){
        if(stream == NULL)
            XMemCopy(t->data, t->devID, s->data, s->devID, s->unitSize * s->unitNum);
        else
            XMemCopyAsync(t->data, t->devID, s->data, s->devID, s->unitSize * s->unitNum, stream->stream, stream->devID);
    }
    /* dense -> sparse */
    else if(!s->isSparse && t->isSparse && 
            s->dataType == DTYPE_IN_MATRIX && 
            t->dataType == DTYPE_IN_MATRIX)
    {
        ShowNiuTransErrors("TODO!");
    }
    /* sparse -> dense */
    else if(s->isSparse && !t->isSparse && 
            s->dataType == DTYPE_IN_MATRIX && 
            t->dataType == DTYPE_IN_MATRIX)
    {
       ShowNiuTransErrors("TODO!");
    }
    /* sparse -> sparse */
    else if(s->isSparse && t->isSparse && 
            s->dataType == DTYPE_IN_MATRIX && 
            t->dataType == DTYPE_IN_MATRIX)
    {
        int num = s->GetNonzeroSize();
        int size = sizeof(int) + num * (s->unitSize + sizeof(int));

        if(stream == NULL)
            XMemCopy(t->data, t->devID, s->data, s->devID, size);
        else
            XMemCopyAsync(t->data, t->devID, s->data, s->devID, size, stream->stream, stream->devID);

        t->unitNumNonZero = num;
    }
    else{
        ShowNiuTransErrors("TODO!");
    }

    return true;
}

/*
flush a list of XTensor to GPU memory
>> mList - list of the tensors
>> GPUMem - memory pool for the GPU
*/
void CudaCPUToGPUFlush(List * mList, XMem * GPUMem)
{
    if (mList == NULL || mList->count == 0)
        return;

#ifdef USE_CUDA
    int size = 0, p = 0;
    int reqiredSize = 0;

    /* compute the requried memory size */
    for (int i = 0; i < mList->count; i++) {
        XTensor * m = (XTensor*)mList->GetItem(i);

        CheckNiuTransErrors((m->devID < 0), "Cannot do gpu-flush on matrices that are already on GPUs.");

        if (m->isSparse)
            reqiredSize = sizeof(int) + (sizeof(int) + m->unitSize) * m->unitNumNonZero;
        else
            reqiredSize = m->unitSize * m->unitNum;

        //reqiredSize = (int)GPUMem->GetPitch(GPUMem->devID, (MTYPE)GPUMem->GetAddress() + size, reqiredSize);
        size += reqiredSize;
    }

    char * data = new char[size];
    char * GPUData = (char*)GPUMem->Alloc(GPUMem->devID, size);
    int pSize = 0;

    /* place the data in a memory block */
    for (int i = 0; i < mList->count; i++) {
        XTensor * m = (XTensor*)mList->GetItem(i);

        if (m->isSparse)
            pSize = sizeof(int) + (sizeof(int) + m->unitSize) * m->unitNumNonZero;
        else
            pSize = m->unitSize * m->unitNum;

        //reqiredSize = (int)GPUMem->GetPitch(GPUMem->devID, (MTYPE)GPUMem->GetAddress() + p, pSize);
        reqiredSize = pSize;

        memcpy(data + p, m->data, pSize);

        if (m->dataHost != NULL)
            delete[](char*)m->dataHost;

        m->dataHost = NULL;
        m->data = GPUData + p;
        m->devID = GPUMem->devID;
        m->mem = GPUMem;

        p += reqiredSize;
    }

    /* copy from CPU memory to GPU memory */
    cudaMemcpy(GPUData, data, size, cudaMemcpyHostToDevice);

    delete[] data;
#endif
}

/* copy the data from GPU memory to CPU memory */
void CudaGPUToCPUFlush(XTensor * tensor)
{
    CheckNiuTransErrors((sizeof(DTYPE) == tensor->unitSize), "Unsupported data type.");

    if (tensor->dataHost != NULL)
        delete[](char*)tensor->dataHost;

    if (tensor->isSparse) {
        int num = int(tensor->unitNum * tensor->denseRatio + 1);
        cudaMemcpy(&num, (DTYPE*)tensor->data, sizeof(int), cudaMemcpyDeviceToHost);

        int tupleSize = sizeof(int) + sizeof(DTYPE);
        int size = sizeof(int) + tupleSize*(num);

        CheckNiuTransErrors((size >= 0), "Illegal data size in the sparse matrix!");

        tensor->dataHost = new char[size];
        cudaMemcpy(tensor->dataHost, tensor->data, size, cudaMemcpyDeviceToHost);
    }
    else {
        tensor->dataHost = new char[tensor->unitNum * tensor->unitSize];
        if (tensor->data != NULL)
            cudaMemcpy(tensor->dataHost, tensor->data, tensor->unitNum * tensor->unitSize, cudaMemcpyDeviceToHost);
        else
            memset(tensor->dataHost, 0, tensor->unitNum * tensor->unitSize);
    }
}

/* 
set the cell to the ascending order along a given dimension (kernel code)
>> data - the data array
>> stride - how many items we go ove when move to the next item along the dimension
>> strideNum - size of the given dimension
>> blockNum - block number
*/
__global__
void KernelSetAscendingOrder(int * data, int stride, int strideNum, int blockNum)
{
    __shared__ int iBlock[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int iOffset[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* index along the "stride" dimension */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* index along the leading dimension */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= stride * blockNum || j >= strideNum)
        return;

    if(threadIdx.y == 0){
        iBlock[threadIdx.x] = i / stride;
        iOffset[threadIdx.x] = i % stride;
    }
    __syncthreads();
    
    int * d = (int*)data + (iBlock[threadIdx.x] * strideNum + j) * stride + iOffset[threadIdx.x];
    *d = j;
}

/* 
set the cell to the ascending order along a given dimension
>> a - the tensor
>> dim - the dimension
*/
extern "C"
void CudaSetAscendingOrder(XTensor * a, int dim)
{
    CheckNiuTransErrors((a->dataType == X_INT), "TODO!");

    int stride = 1;
    int strideNum = a->dimSize[dim];
    for(int i = 0; i < dim; i++)
        stride *= a->dimSize[i];

    int blockNum = 1;
    for(int i = dim + 1; i < a->order; i++)
        blockNum *= a->dimSize[i];

    int gridSize[3];
    int blockSize[3];

    GDevs->GetGridAndBlockSize2D(a->devID, strideNum, stride * blockNum, MAX_INT, gridSize, blockSize);

    KernelSetAscendingOrder<<<dim3(gridSize[1], gridSize[0]), dim3(blockSize[1], blockSize[0])>>>
                             ((int*)a->data, stride, strideNum, blockNum);
}

/* 
set each entry to its negtive value (CUDA Kernel) 
>> d - pointer to the data array
>> size - size of the data array
*/
__global__ 
void KernelNegate(DTYPE * d, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        d[i] = -d[i];
}

/* 
set each entry to its negtive value (CUDA Kernel) 
This is for float16 computation
>> d - pointer to the data array
>> size - size of the data array
*/
__global__ 
void KernelNegate(__half * d, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    if(i < size)
        d[i] = __hsub(__float2half(0), d[i]);
#else
    if(i < size)
        d[i] = __float2half(-__half2float(d[i]));
#endif
}

/* 
set each entry to its negtive value 
>> a - the tensor
*/
extern "C"
void CudaNegate(XTensor * a)
{
    CheckNiuTransErrors((a->isSparse == false), "TODO!");

    int gridSize[3];
    int blockSize[3];

    GDevs->GetGridAndBlockSize(a->devID, a->unitNum, gridSize, blockSize);

    dim3 blocks(gridSize[0]);
    dim3 threads(blockSize[0]);

    if(a->dataType == DTYPE_IN_MATRIX){
        KernelNegate<<<blocks, threads>>>((DTYPE*)a->data, a->unitNum);
    }
    else if(a->dataType == X_FLOAT16){
        KernelNegate<<<blocks, threads>>>((__half*)a->data, a->unitNum);
    }
    else{
        ShowNiuTransErrors("TODO!");
    }
}

/* 
set all entries to its root (CUDA Kernel) 
>> d - data array
>> size - size of the data array
*/
__global__
void KernelSqrtV2(DTYPE * d, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        d[i] = sqrt(d[i]);
}

/* 
set all entries to its root (CUDA Kernel) 
>> d - data array
>> size - size of the data array
*/
__global__
void KernelSqrtV2(__half * d, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    if (i < size)
        d[i] = hsqrt(d[i]);
#else
    if (i < size)
        d[i] = __float2half(sqrt(__half2float(d[i])));
#endif
}

/* 
get power(d[i], p)
>> d - data array
>> p - power
>> size - size of the data array
*/
__global__
void KernelPower(DTYPE * d, DTYPE p, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        d[i] = pow(d[i], p);
}

/* 
get power(d[i], p)
>> d - data array
>> p - power
>> size - size of the data array
*/
__global__
void KernelPower(__half * d, __half p, int size)
{
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    //int i = blockDim.x * blockIdx.x + threadIdx.x;
    //if (i < size)
    //    d[i] = hpow(d[i], p);
#else
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        d[i] = __float2half(pow(__half2float(d[i]), __half2float(p)));
#endif
}

/* get the power of the entries */
extern "C"
void CudaPower(XTensor * a, DTYPE p)
{
    int gridSize[3];
    int blockSize[3];

    GDevs->GetGridAndBlockSize(a->devID, a->unitNum, gridSize, blockSize);

    dim3 blocks(gridSize[0]);
    dim3 threads(blockSize[0]);

    if(a->dataType == DTYPE_IN_MATRIX){
        if(p == (DTYPE)0.5){
            KernelSqrtV2<<<blocks, threads>>>((DTYPE*)a->data, a->unitNum);
        }
        else if(p != (DTYPE)1.0){
            KernelPower<<<blocks, threads>>>((DTYPE*)a->data, p, a->unitNum);
        }
    }
    else if(a->dataType == X_FLOAT16){
        if(p == (DTYPE)0.5){
            KernelSqrtV2<<<blocks, threads>>>((__half*)a->data, a->unitNum);
        }
        else if(p != (DTYPE)1.0){
            ShowNiuTransErrors("TODO!");
            //unsigned short p2 = FloatToFloat16(p);
            //__half * pp = (__half*)&p2;
            //KernelPower<<<blocks, threads>>>((__half*)a->data, *pp, a->unitNum);
        }
    }
    else{
        ShowNiuTransErrors("TODO!");
    }
}

/* 
scale and shift all matrix entires p = p * scale + shift (CUDA Kernel) 
>> d - the data array
>> size - the size of d
>> scale - how much we want to scale it
>> shift - how much we want to shift it
*/
__global__ 
void KernelScaleAndShift(DTYPE * d, int size, DTYPE scale, DTYPE shift)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        d[i] = d[i] * scale + shift;
}

/* 
scale and shift all matrix entires p = p * scale + shift (CUDA Kernel) 
This is for float16 computation
>> d - the data array
>> size - the size of d
>> scale - how much we want to scale it
>> shift - how much we want to shift it
*/
__global__ 
void KernelScaleAndShift(__half * d, int size, __half scale, __half shift)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    if(i < size)
        d[i] = __hadd(__hmul(d[i], scale), shift);
#else
    if (i < size)
        d[i] = __float2half(__half2float(d[i]) * __half2float(scale) + __half2float(shift));
#endif
}

/* 
scale and shift all matrix entires

p = p * scale + shift

>> a - the matrix
>> scale - the scaler factor
>> shift - the shift factor
*/
void CudaScaleAndShift(XTensor * a, DTYPE scale, DTYPE shift)
{
    /* sparse matrix */
    if(a->isSparse){
        // TODO
    }
    /* dense matrix */
    else{
        int gridSize[3];
        int blockSize[3];

        GDevs->GetGridAndBlockSize(a->devID, a->unitNum, gridSize, blockSize);

        dim3 blocks(gridSize[0]);
        dim3 threads(blockSize[0]);

        if(a->dataType == DTYPE_IN_MATRIX){
            KernelScaleAndShift<<<blocks, threads>>>((DTYPE*)a->data, a->unitNum, scale, shift);
        }
        else if(a->dataType == X_FLOAT16){
            unsigned short scale2 = FloatToFloat16(scale);
            unsigned short shift2 = FloatToFloat16(shift);
            __half * scaleft16p = (__half*)&scale2;
            __half * shiftft16p = (__half*)&shift2;
            KernelScaleAndShift<<<blocks, threads>>>((__half*)a->data, a->unitNum, *scaleft16p, *shiftft16p);
        }
        else{
            ShowNiuTransErrors("TODO!");
        }
    }
}

/* 
copy a number of blocks to target positions
NOTE that this version makes more use of the 2d threads in cuda
>> source - data array (head of the blocks) to copy from
>> blockSize - size of block
>> blockNum - number of blocks
>> target - target data array
>> targetBlocks - target positions of the copy
*/
template<int miniBlockSize>
__global__ 
void KernelCopyBlocks(DTYPE * source, int blockSize, int blockNum, DTYPE * target, int * targetBlocks)
{
    /* entry index in the block */
    int i = (blockDim.x * blockIdx.x + threadIdx.x) * miniBlockSize;

    /* block index */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(j >= blockNum)
        return;
    
    /* target position */
    int k = targetBlocks[j];

    DTYPE * s = source + blockSize * j;
    DTYPE * t = target + blockSize * k;

    if(i < blockSize){
        if(miniBlockSize == 4){
            t[i] = s[i];
            t[i + 1] = s[i + 1];
            t[i + 2] = s[i + 2];
            t[i + 3] = s[i + 3];
        }
        else if(miniBlockSize <= 1){
            t[i] = s[i];
        }
        else{
            printf("something wrong!");
        }
    }
}

/* 
copy a number of blocks to target positions (cuda version)
>> source - data array (head of the blocks) to copy from
>> blockSize - size of block
>> blockNum - number of blocks
>> target - target data array
>> targetBlocks - target positions of the copy (on the device)
>> myMem - memory pool
*/
void CudaCopyBlocks(void * source, int blockSize, int blockNum, void * target, int * targetBlocks, XMem * myMem)
{
    CheckNiuTransErrors((myMem != NULL), "No memory pool!");
    CheckNiuTransErrors((myMem->devID >= 0), "Wrong device to run!");
    CheckNiuTransErrors((blockSize % sizeof(DTYPE) == 0), "Unsupported block size!");

    int cudaGrids[3];
    int cudaBlocks[3];
    int bSize = blockSize/sizeof(DTYPE);

    if(bSize % 4 == 0){
        GDevs->GetGridAndBlockSize2D(myMem->devID, bSize/4, blockNum, MAX_INT, cudaGrids, cudaBlocks);
        KernelCopyBlocks<4> <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                              ((DTYPE*)source, bSize, blockNum, (DTYPE*)target, targetBlocks);
    }
    else{
        GDevs->GetGridAndBlockSize2D(myMem->devID, bSize, blockNum, MAX_INT, cudaGrids, cudaBlocks);
        KernelCopyBlocks<1> <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                                  ((DTYPE*)source, bSize, blockNum, (DTYPE*)target, targetBlocks);
    }
}

/* 
copy a number of blocks from source positions to target positions
>> source - data array (head of the blocks) to copy from
>> blockSize - size of block
>> sourceBlocks - source positions of the copy
>> blockNum - number of blocks
>> target - target data array
>> targetBlocks - target positions of the copy
*/
__global__ 
void KernelCopyBlocksSelected(DTYPE * source, int blockSize, int * sourceBlocks, int blockNum, DTYPE * target, int * targetBlocks)
{
    /* block index */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* entry index in the block */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(j >= blockNum)
        return;
    
    /* target position */
    int srcIndex = sourceBlocks[j];
    int tgtIndex = targetBlocks[j];

    DTYPE * s = source + blockSize * srcIndex;
    DTYPE * t = target + blockSize * tgtIndex;

    if(i < blockSize)
        t[i] = s[i];
}

/* 
copy a number of blocks from source positions to target positions (cuda version)
>> source - data array (head of the blocks) to copy from
>> blockSize - size of block
>> sourceBlocks - source positions of the copy
>> blockNum - number of blocks
>> target - target data array
>> targetBlocks - target positions of the copy
>> myMem - memory pool
*/
void CudaCopyBlocksSelected(void * source, int blockSize, int * sourceBlocks, int blockNum, void * target, int * targetBlocks, XMem * myMem)
{
    CheckNiuTransErrors((myMem != NULL), "No memory pool!");
    CheckNiuTransErrors((myMem->devID >= 0), "Wrong device to run!");
    CheckNiuTransErrors((blockSize % sizeof(DTYPE) == 0), "Unsupported block size!");

    /* copy the index to the GPU memory */
    int * sourceBlocksTMP = (int*)myMem->AllocBuf(myMem->devID, blockNum * sizeof(int));
    int * targetBlocksTMP = (int*)myMem->AllocBuf(myMem->devID, blockNum * sizeof(int));
    XMemCopy(sourceBlocksTMP, myMem->devID, sourceBlocks, -1, blockNum * sizeof(int));
    XMemCopy(targetBlocksTMP, myMem->devID, targetBlocks, -1, blockNum * sizeof(int));

    int cudaGrids[3];
    int cudaBlocks[3];

    GDevs->GetGridAndBlockSize2D(myMem->devID, blockSize/sizeof(DTYPE), blockNum, MAX_INT, cudaGrids, cudaBlocks);

    KernelCopyBlocksSelected<<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                            ((DTYPE*)source, blockSize/sizeof(DTYPE), sourceBlocksTMP, blockNum, (DTYPE*)target, targetBlocksTMP);

    myMem->ReleaseBuf(myMem->devID, blockNum * sizeof(int));
    myMem->ReleaseBuf(myMem->devID, blockNum * sizeof(int));
}

/* 
set target data block index for the data movement in split (device code)
>> blockIndex - block index
>> splitNum - number of splits
>> blockSplitSize - size of the splitted block
>> blockNum - number of data blocks
*/
__global__
void KernelMakeSplitBlockIndex(int * blockIndex, int splitNum, int blockSplitSize, int blockNum)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i >= blockNum)
        return;

    int j = (i % splitNum) * blockSplitSize + i / splitNum;

    /* i = source block index, j = target block index */
    blockIndex[i] = j;
}

/* 
set target data block index for the data movement in split
>> blockIndex - block index
>> splitNum - number of splits
>> blockSplitSize - size of the splitted block
>> blockNum - number of data blocks
*/
extern "C" 
void CudaMakeSplitBlockIndex(int devID, int * blockIndex, int splitNum, int blockSplitSize, int blockNum)
{
    int cudaGrids[3];
    int cudaBlocks[3];

    GDevs->GetGridAndBlockSize(devID, blockNum, cudaGrids, cudaBlocks);

    KernelMakeSplitBlockIndex<<<dim3(cudaGrids[0]), dim3(cudaBlocks[0])>>>
                                (blockIndex, splitNum, blockSplitSize, blockNum);
}

/* 
copy a number of blocks (of different sizes) to target positions
>> sourceList - list of data arrays to copy from
>> sourceBlockSizes - the size of the block_i
>> sourceBlockNum - number of blocks to merge
>> targetList - list of data arrays to copy to
>> target - target data array
*/
__global__ 
void KernelCopyBlockLists(DTYPE * sourceList[], int * sourceBlockSizes, int sourceBlockNum, DTYPE * targetList[])
{
    __shared__ int iBlockSizes[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE * iSourceList[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE * iTargetList[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* entry index in the block */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* block index */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(j >= sourceBlockNum)
        return;

    if(threadIdx.x == 0){
        iBlockSizes[threadIdx.y] = sourceBlockSizes[j];
        iSourceList[threadIdx.y] = sourceList[j];
        iTargetList[threadIdx.y] = targetList[j];
    }

    __syncthreads();

    if(i < iBlockSizes[threadIdx.y])
        iTargetList[threadIdx.y][i] = iSourceList[threadIdx.y][i];
}

/*
merge data by blocks (cuda version)
>> sourceList - list of data arrays (heads of the blocks) to copy from
>> blockSizes - size of the blocks
>> blockNum - number of blocks
>> target - target data array
>> myMem - the memory pool
*/
extern "C" 
void CudaMergeBlockLists(List * sourceList, int * blockSizes, int blockNum, void * target, XMem * myMem)
{
    CheckNiuTransErrors((myMem != NULL), "No memory pool!");
    CheckNiuTransErrors((myMem->devID >= 0), "Wrong device to run!");

    int newBlockListSize = sourceList->count * blockNum;

    int minBlockSize = MAX_INT;
    int maxBlockSize = -MAX_INT;
    //int realMinBlockSize = 1;
    int realMaxBlockSize = 1;
    DTYPE ** sourceArrays = new DTYPE*[newBlockListSize];
    DTYPE ** targetArrays = new DTYPE*[newBlockListSize];
    int * sizes = new int[newBlockListSize];
    int * offsets = new int[sourceList->count];
    memset(offsets, 0, sizeof(int) * sourceList->count);

    int totalOffset = 0;
    for (int k = 0; k < blockNum; k++){
        for (int i = 0; i < sourceList->count; i++){
            CheckNiuTransErrors((blockSizes[i] % sizeof(DTYPE) == 0), "Unsupported block size!");
            int j = k * sourceList->count + i;
            sizes[j] = blockSizes[i] / sizeof(DTYPE);
            sourceArrays[j] = (DTYPE*)sourceList->GetItem(i) + offsets[i];
            targetArrays[j] = (DTYPE*)target + totalOffset;
            offsets[i] += sizes[i];
            totalOffset += sizes[i];

            if (minBlockSize > blockSizes[i])
                minBlockSize = blockSizes[i];
            if(maxBlockSize < blockSizes[i])
                maxBlockSize = blockSizes[i];
        }
    }

    CheckNiuTransErrors((minBlockSize % sizeof(DTYPE) == 0), "Unsupported block size!");
    CheckNiuTransErrors((maxBlockSize % sizeof(DTYPE) == 0), "Unsupported block size!");
    //realMinBlockSize = minBlockSize/sizeof(DTYPE);
    realMaxBlockSize = maxBlockSize/sizeof(DTYPE);
    
    int cudaGridSizes[3];
    int cudaBlockSizes[3];

    GDevs->GetGridAndBlockSize2D(myMem->devID, realMaxBlockSize, newBlockListSize, MAX_INT, 
                                 cudaGridSizes, cudaBlockSizes);

    myMem->SetPinBuf();
    //MTYPE offset0 = myMem->bufUsed;
    int * sizesGPU = (int*)myMem->AllocBuf(myMem->devID, sizeof(int) * newBlockListSize, 256);
    
    //MTYPE offset1 = myMem->bufUsed;
    DTYPE ** sourceArraysGPU = (DTYPE**)myMem->AllocBuf(myMem->devID, sizeof(DTYPE*) * newBlockListSize, 256);
    
    //MTYPE offset2 = myMem->bufUsed;
    DTYPE ** targetArraysGPU = (DTYPE**)myMem->AllocBuf(myMem->devID, sizeof(DTYPE*) * newBlockListSize, 256);
    
    //MTYPE bufSize = myMem->bufUsed - offset0;

    //char * CPUBuf = new char[bufSize];
    //memset(CPUBuf, 0 , bufSize);

    //memcpy(CPUBuf, sizes, sizeof(int) * newBlockListSize);
    //memcpy(CPUBuf + (offset1 - offset0), sourceArrays, sizeof(DTYPE*) * newBlockListSize);
    //memcpy(CPUBuf + (offset2 - offset0), targetArrays, sizeof(DTYPE*) * newBlockListSize);

    XMemCopy(sizesGPU, myMem->devID, sizes, -1, sizeof(int) * newBlockListSize);
    XMemCopy(sourceArraysGPU, myMem->devID, sourceArrays, -1, sizeof(DTYPE*) * newBlockListSize);
    XMemCopy(targetArraysGPU, myMem->devID, targetArrays, -1, sizeof(DTYPE*) * newBlockListSize);

    /* it is VERY tricky here because we squeeze three data copies into one */
    //XMemCopy(sizesGPU, myMem->devID, CPUBuf, -1, bufSize);

    KernelCopyBlockLists<<<dim3(cudaGridSizes[0],cudaGridSizes[1]), dim3(cudaBlockSizes[0],cudaBlockSizes[1])>>>
                          (sourceArraysGPU, sizesGPU, newBlockListSize, targetArraysGPU);

    myMem->BackToPinBuf();

    delete[] sourceArrays;
    delete[] targetArrays;
    delete[] sizes;
    delete[] offsets;
    //delete[] CPUBuf;
}

/* 
set target data block index for the data movement in split (device code)
>> blockIndex - index of the blocks
>> blockNum - number of the blocks
>> blockNumInMerge - size of the dimension along which we perform the merging operation
>> splitSizeInGrid - size of each data array to merge 
>> gridSize - number of blocks in a grid (here grid is a higher level orgnization upon blocks)
>> gridNum - number of grids
>> mem - the memory pool
*/
__global__
void KernelMakeMergeBlockIndex(int * blockIndex, int blockNum, int blockNumInMerge, 
                               int splitSizeInGrid, int gridSize, int gridNum)
{
    /* block index */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* grid index */
    int k = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= blockNum || k >= gridNum)
        return;

    int j = blockNumInMerge * (i % splitSizeInGrid) + int(i / splitSizeInGrid);

    /* i = source block index, j = target block index and k = (target) grid index */
    blockIndex[i + gridSize * k] = j + gridSize * k;
}

/* 
set target data block index for the data movement in split 
>> devID - id of the GPU device
>> blockIndex - index of the blocks
>> blockNum - number of the blocks
>> blockNumInMerge - size of the dimension along which we perform the merging operation
>> splitSizeInGrid - size of each data array to merge 
>> gridSize - number of blocks in a grid (here grid is a higher level orgnization upon blocks)
>> gridNum - number of grids
>> mem - the memory pool
*/
extern "C" 
void CudaMakeMergeBlockIndex(int devID, 
                             int * blockIndex, int blockNum, int blockNumInMerge, 
                             int splitSizeInGrid, int gridSize, int gridNum)
{
    int cudaGrids[3];
    int cudaBlocks[3];

    GDevs->GetGridAndBlockSize2D(devID, blockNum, gridNum, MAX_INT, cudaGrids, cudaBlocks);

    KernelMakeMergeBlockIndex<<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                               (blockIndex, blockNum, blockNumInMerge, splitSizeInGrid, gridSize, gridNum);
}

/* 
insert a dimension by copying the blocks for n times (where n is the size of the inerted dimension) 
>> s - pointer to the source data array
>> blockSize - size of a block
>> blockNum - number of the blocks
>> t - pointer to the target data array
*/
template<class T>
__global__
void KernelUnsqueeze(void * s, int blockSize, int blockNum, void * t, int n)
{
    /* index of data items */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* block index */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= blockSize || j >= blockNum)
        return;

    MTYPE offset = blockSize * j;
    T value = ((T*)s)[offset + i];
    T * tData = (T*)t + offset * n;
    int length = blockSize * n;

    __syncthreads();

    for(int k = i; k < length; k += blockSize)
        tData[k] = value;
}

/* 
insert a dimension by copying the blocks for x times (where x is the size of the inerted dimension) 
>> a - input tensor
>> b - output tensor
>> dim - where to insert the dimension
>> dSize - size of the newly-inserted dimension
*/
extern "C"
void CudaUnsqueeze(XTensor * a, XTensor * b, int dim, int dSize)
{
    int blockSize = 1;
    int blockNumA = 1;
    int blockNumB = 1;
    for(int i = 0; i < dim; i++)
        blockSize *= a->dimSize[i];

    blockNumA = a->unitNum / blockSize;
    blockNumB = b->unitNum / blockSize;

    CheckNiuTransErrors((blockNumA * dSize == blockNumB), "Unmatched tensors!");;

    int cudaGrids[3];
    int cudaBlocks[3];

    GDevs->GetGridAndBlockSize2D(a->devID, blockSize, blockNumA, MAX_INT, cudaGrids, cudaBlocks);

    if(a->dataType == X_FLOAT && a->dataType == X_FLOAT){
        KernelUnsqueeze<float> <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                                 (a->data, blockSize, blockNumA, b->data, dSize);
    }
    else if(a->dataType == X_INT && a->dataType == X_INT){
        KernelUnsqueeze<int>   <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                                 (a->data, blockSize, blockNumA, b->data, dSize);
    }
    else{
        ShowNiuTransErrors("TODO!");
    }
}

/*
set the data arrry with a default value
>> data - data array
>> value - default value
>> size - size of the array
*/
template<class T> __global__ 
void KernelSetDataArray(T * data, T value, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < size)
        data[i] = value;
}

/* 
reorganize data blocks (in a tensor) into a matrix. In each (source) block
we have stride * strideNum items, where strideNum means the items along the
leading dimension. In the target matrix, each row keeps strideNum items along
the leading dimension in each source block.
>> source - source data array
>> target - target data array
>> srcStride - how many items we need to go over we move to the next
>> srcStrideNum - size of the leading dimension
>> srcBlockNum - number of the source blocks
>> tgtColNum - number of columns in the target matrix
>> tgtRowNum - number of rows in the target matrix
*/
template<class T> __global__ 
void KernelReorganize(void * source, void * target, 
                      int srcStride, int srcStrideNum, int srcBlockNum, 
                      int tgtColNum, int tgtRowNum)
{
    __shared__ int iBlock[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int iOffset[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* index along the "stride" dimension */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* index along the leading dimension */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= srcStride * srcBlockNum || j >= srcStrideNum)
        return;

    if(threadIdx.y == 0){
        iBlock[threadIdx.x] = i / srcStride;
        iOffset[threadIdx.x] = i % srcStride;
    }
    __syncthreads();

    T * s = (T*)source + (iBlock[threadIdx.x] * srcStrideNum + j) * srcStride + iOffset[threadIdx.x];
    T * t = (T*)target + (iBlock[threadIdx.x] * srcStride + iOffset[threadIdx.x]) * tgtColNum + j;
    *t = *s;
}

/*
copy back for "KernelReorganize"
>> source - source data array
>> target - target data array
>> srcColNum - number of columns in the source matrix
>> srcRowNum - number of rows in the source matrix
>> tgtStride - how many items we need to go over we move to the next
>> tgtStrideNum - size of the leading dimension
>> tgtBlockNum - number of the target blocks
*/
template<class T> __global__ 
void KernelReorganizeBack(void * source, void * target, 
                          int srcColNum, int srcRowNum,
                          int tgtStride, int tgtStrideNum, int tgtBlockNum)
{
    __shared__ int iBlock[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int iOffset[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* index along the "stride" dimension */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* index along the leading dimension */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= tgtStride * tgtBlockNum || j >= tgtStrideNum)
        return;

    if(threadIdx.y == 0){
        iBlock[threadIdx.x] = i / tgtStride;
        iOffset[threadIdx.x] = i % tgtStride;
    }
    __syncthreads();

    T * s = (T*)source + (iBlock[threadIdx.x] * tgtStride + iOffset[threadIdx.x]) * srcColNum + j;
    T * t = (T*)target + (iBlock[threadIdx.x] * tgtStrideNum + j) * tgtStride + iOffset[threadIdx.x];
    *t = *s;
}

/* 
bitonic sort (for each row in a matrix)
>> data - pointer to the data array
>> index - index data array
>> j - segment/distance for comparsion
>> k - length of the monotonic sequence
>> m - column number of the matrix
>> n - row number of the matrix
*/
template<class T> __global__
void KernelBitonicSort2D(void * data, int j, int k, int m, int n)
{
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx >= m || row >= n)
        return;

    T * items = (T*)data + m * row;

    int ixj = idx^j;
    if(ixj > idx){
        if((idx&k) == 0 && items[idx] < items[ixj]){
            T tmp = items[idx];
            items[idx] = items[ixj];
            items[ixj] = tmp;
        }
        if((idx&k) != 0 && items[idx] > items[ixj]){
            T tmp = items[idx];
            items[idx] = items[ixj];
            items[ixj] = tmp;
        }
    }
}

/* 
bitonic sort (for each row in a matrix) with index
>> data - pointer to the data array
>> index - index data array
>> j - segment/distance for comparsion
>> k - length of the monotonic sequence
>> m - column number of the matrix
>> n - row number of the matrix
*/
template<class T> __global__
void KernelBitonicSort2D(void * data, int * index, int j, int k, int m, int n)
{
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx >= m || row >= n)
        return;

    T * items = (T*)data + m * row;
    int * indexOnSite = index + m * row;

    int ixj = idx^j;
    if(ixj > idx){
        if((idx&k) == 0 && items[idx] < items[ixj]){
            T tmp = items[idx];
            items[idx] = items[ixj];
            items[ixj] = tmp;
            int tmp2 = indexOnSite[idx];
            indexOnSite[idx] = indexOnSite[ixj];
            indexOnSite[ixj] = tmp2;
        }
        if((idx&k) != 0 && items[idx] > items[ixj]){
            T tmp = items[idx];
            items[idx] = items[ixj];
            items[ixj] = tmp;
            int tmp2 = indexOnSite[idx];
            indexOnSite[idx] = indexOnSite[ixj];
            indexOnSite[ixj] = tmp2;
        }
    }
}


/* 
sort the tensor along a given dimension
>> a - input
>> b - output
>> indexA - input index tensor
>> indexB - output index tensor
>> dim - specified dimension
>> k - top-k results are returned
*/
void CudaSortBig(XTensor * a, XTensor * b, XTensor * indexA, XTensor * indexB, int dim, int k)
{
    CheckNiuTransErrors((a && b), "Empty input tensor!");
    CheckNiuTransErrors((a->unitSize == b->unitSize), "Unmatched tensors!");
    CheckNiuTransErrors((a->order > dim && dim >= 0), "Incorrect dimension specified!");
    CheckNiuTransErrors((a->dataType == DTYPE_IN_MATRIX), "TODO!");

    if(k < 0 || k > b->dimSize[dim])
        k = b->dimSize[dim];
    
    XMem * mem = a->mem;

    int stride = 1;
    int strideNum = a->dimSize[dim];
    for(int i = 0; i < dim; i++)
        stride *= a->dimSize[i];

    int blockNum = 1;
    for(int i = dim + 1; i < a->order; i++)
        blockNum *= a->dimSize[i];

    int m = GetNextPower2(strideNum);
    int n = stride * blockNum;
    
    void * buf = mem->AllocBuf(mem->devID, n * m * a->unitSize);
    void * bufIndex = (indexA != NULL && indexB != NULL) ? mem->AllocBuf(mem->devID, n * m * sizeof(int)) : NULL;

    int cudaGrids[3];
    int cudaBlocks[3];

    GDevs->GetGridAndBlockSize(mem->devID, m * n, cudaGrids, cudaBlocks);

    /* set the buffer to the "min" value */
    KernelSetDataArray<DTYPE> <<<dim3(cudaGrids[0]), dim3(cudaBlocks[0])>>>
                                ((DTYPE*)buf, DTYPE_MIN, m * n);

    GDevs->GetGridAndBlockSize2D(mem->devID, strideNum, n, MAX_INT, cudaGrids, cudaBlocks);

    /* reorganize the data into a matrix */
    KernelReorganize<DTYPE> <<<dim3(cudaGrids[1], cudaGrids[0]), dim3(cudaBlocks[1], cudaBlocks[0])>>>
                               (a->data, buf, stride, strideNum, blockNum, m, n);

    /* reorganize the index into a matrix */
    if(indexA != NULL && indexB != NULL)
        KernelReorganize<int> <<<dim3(cudaGrids[1], cudaGrids[0]), dim3(cudaBlocks[1], cudaBlocks[0])>>>
                                (indexA->data, bufIndex, stride, strideNum, blockNum, m, n);

    GDevs->GetGridAndBlockSize2D(mem->devID, m, n, MAX_INT, cudaGrids, cudaBlocks);

    /* bitonic sorting */
    for(int i = 2; i <= m; i <<= 1){
        for(int j = i >> 1; j > 0; j = j >> 1){
            if(indexA != NULL && indexB != NULL){
                KernelBitonicSort2D<DTYPE> <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                                             (buf, (int*)bufIndex, j, i, m, n);
            }
            else{
                KernelBitonicSort2D<DTYPE> <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                                             (buf, j, i, m, n);
            }
        }
    }

    GDevs->GetGridAndBlockSize2D(mem->devID, k, n, MAX_INT, cudaGrids, cudaBlocks);

    /* copy result to the output tensor */
    KernelReorganizeBack<DTYPE> <<<dim3(cudaGrids[1], cudaGrids[0]), dim3(cudaBlocks[1], cudaBlocks[0])>>>
                                  (buf, b->data, m, n, stride, k, blockNum);

    if(indexA != NULL && indexB != NULL)
        KernelReorganizeBack<int> <<<dim3(cudaGrids[1], cudaGrids[0]), dim3(cudaBlocks[1], cudaBlocks[0])>>>
                                    (bufIndex, indexB->data, m, n, stride, k, blockNum);

    mem->ReleaseBuf(mem->devID, n * m * a->unitSize);
    if(indexA != NULL && indexB != NULL)
        mem->ReleaseBuf(mem->devID, n * m * sizeof(int));
}

/**************************************/

/* heap item */
template <typename T>
struct CudaHeapNode
{
    /* node index */
    int index;

    /* value of the node */
    T value;

    __device__ CudaHeapNode(){};

    __device__ CudaHeapNode(int i, T v)
    {
        index = i;
        value = v;
    };
};

/* heap (device code) */
template<HeapType hType, typename T>
class CudaXHeap
{
public:
    /* number of the items the heap keeps */
    int size;

    /* number of the items that are already in the heap */
    int count;

    /* items */
    CudaHeapNode<T> * items;

    /* value for the top-most item*/
    T topValue;

public:
    /* constructor */
    __device__ CudaXHeap(int mySize, CudaHeapNode<T> * myItems)
    {
        size = mySize;
        count = 0;
        items = myItems;
        topValue = 0;
    }
    /* constructor */
    __device__ CudaXHeap(int mySize, int myCount, CudaHeapNode<T> * myItems)
    {
        size = mySize;
        count = myCount;
        items = myItems;
        topValue = items[0].value;
    }
    /* compare node i and node j */
    __device__ bool Compare(int i, int j)
    {
        if (hType == MIN_HEAP)
            return items[i].value < items[j].value;
        else
            return items[j].value < items[i].value;
    }

    /* swap */
    __device__ void Swap(int i, int j)
    {
        /*CudaHeapNode<T> tmp = items[i];
        items[i] = items[j];
        items[j] = tmp;*/
        int tmpIndex = items[i].index;
        T tmpValue = items[i].value;
        items[i] = items[j];
        items[j].index = tmpIndex;
        items[j].value = tmpValue;
    }

    /* replace the top-most item and update the heap */
    __device__ void ReplaceTop(CudaHeapNode<T> node)
    {
        items[0] = node;
        Down(0);
        topValue = items[0].value;
    }

    /* replace the top-most item and update the heap */
    __device__ void ReplaceTop(int index, T value)
    {
        items[0].index = index;
        items[0].value = value;
        Down(0);
        topValue = items[0].value;
    }

    /* push an item into the heap */
    __device__ void Push(CudaHeapNode<T> node)
    {
        items[count] = node;
        Up(count);
        count++;
        topValue = items[0].value;
    }

    /* push an item into the heap */
    __device__ void Push(int index, T value)
    {
        items[count].index = index;
        items[count].value = value;
        Up(count);
        count++;
        topValue = items[0].value;
    }

    /* move item k down the tree */
    __device__ void Down(int k)
    {
        int i = k;
        int i2 = i + i;
        while (i2 + 1 < count) {
            int l = i2 + 1;
            int r = i2 + 2;
            int m = (Compare(l, r) || r >= count) ? l : r;
            if (Compare(i, m))
                break;
            Swap(i, m);
            i = m;
            i2 = m << 1;
        }
    }

    /* move item k up the tree */
    __device__ void Up(int k)
    {
        int i = k;
        int parent = (i - 1) >> 1;
        while (i > 0 && !Compare(parent, i)) {
            Swap(parent, i);
            i = parent;
            parent = (i - 1) >> 1;
        }
    }
};
__device__ unsigned convert(float v)
{
	unsigned x = __float_as_int(v);
	unsigned mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
	return (x ^ mask);
}
__device__ float convert(unsigned int v)
{
	float x = __uint_as_float(v);
	return x;
}
__device__ float deconvert(unsigned int v) {
	unsigned int mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

	return __int_as_float(v ^ mask);
}
__global__ void convert2uint(float* input, unsigned int *output, int size)
{
	for (int i = 0; i < size; i++)
	{
		output[i] = convert(input[i]);
	}
}
__device__ void convert2float(unsigned int* input, float* output, int size)
{
	output[0] = convert(input[0]);
}

/* 
get the top-k items
>> input - the input data array
>> stride - number of items we go over when we move to the next item along a given dimension
>> strideNum - size of the given dimension
>> blockNum - number of data blocks
>> k - as it is
>> minValue - min value of an item
>> output - the output data array
>> index - the output index array
*/
/*

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<cstdio>
#include<malloc.h>
#define SIZE 10
__device__ unsigned convert(float v)
{
unsigned x = __float_as_int(v);
unsigned mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
return (x ^ mask);
}
__global__ void convert2uint(float* input, unsigned int *output,int size)
{
for (int i = 0; i < size; i++)
{
output[i] = convert(input[i]);
printf("%u\n", output[i]);
}
}
__global__ void outputDevice()
{
unsigned int ret = convert(5.0f);
printf("%u\n", ret);
ret = convert(4.0f);
printf("%u\n", ret);
ret = convert(3.0f);
printf("%u\n", ret);
}

int main()
{
float *input = (float*)malloc(sizeof(float)*SIZE),*ginput;
unsigned int* output = (unsigned int*)malloc(sizeof(unsigned int)*SIZE), *goutput;
cudaMalloc(&ginput, sizeof(float)*SIZE);
cudaMalloc(&goutput, sizeof(unsigned int)*SIZE);
for (int i = 0; i < SIZE; i++)
{
input[i] = i;
}
cudaMemcpy(ginput, input, SIZE * sizeof(float),cudaMemcpyHostToDevice);
convert2uint<<<1,1>>>(ginput,goutput,SIZE);
cudaMemcpy(output, goutput, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
printf("-------------------------------------------------");
for (int i = 0; i < SIZE; i++)
{
printf("%u\n", output[i]);
}
return 0;
}

*/
//	radixCount(input, stride*strideNum*blockNum, pos_count, mask, mask_desire, desire, stride,strideNum,blockNum);
__device__ void radixCount(unsigned int *data, int limit, int *pos_count,  unsigned int mask, int mask_desire, unsigned int desire,int stride,int strideNum,int blockNum)
{
	/*the idx th thread in one vector */
	
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	//printf("%d\n", idx);
	/* the idy th vector in one tensor */
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int blockIndex = idy / stride; 
	int offsetInBlock = idy% stride; 
	//int indexOffset = blockDim.x;
	//int dataOffset = stride * blockDim.x;
//	printf("idy: %d, stratIndex: %d\n", idy, stride * strideNum * blockIndex + offsetInBlock);
	//if (idx != 3)
	//	return ;
	/*
	for (int i = stride * strideNum * blockIndex + offsetInBlock+idx, j = 0; j < strideNum; j++, i += stride*32)
	{
	printf("Radix Count: %d\n", i);
	}
	*/
	for (int j = idx*stride+ stride * strideNum * blockIndex + offsetInBlock;
		j<  stride * strideNum * blockIndex + offsetInBlock+ stride*strideNum && j<limit;
		j+= stride*32)
	{
		if ((data[j] & mask_desire) == desire)
		{
			if (data[j] & mask)
			{
				pos_count[(idy % 32)*blockDim.x+idx]++;
			}
		}
	//	printf("Radix Count: %d Idx: %d,Idy: %d,end: %d\n", j,idx,idy, stride * strideNum * blockIndex + offsetInBlock + stride*strideNum);
	}
}


__device__ __forceinline__ unsigned getLaneMaskLe() {
	unsigned mask;
	asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
	return mask;
}
__device__ __forceinline__ int getLaneId() {
	int laneId;
	asm("mov.s32 %0, %laneid;" : "=r"(laneId));
	return laneId;
}

//the theard number need be 32 times 
__device__ void gpu_check_warp(int *smem, bool in, int *carry, int *index)
{
	int vote = __ballot_sync(0xffffffff, in);
	*index = __popc(getLaneMaskLe() & vote);
	*carry = __popc(vote);
	int warp = threadIdx.x / 32;  //get the warp number

	if (getLaneId() == 0)
	{
		smem[warp] = *carry; //save each warp carry
	}
	__syncthreads();
	if (getLaneId() == 0)
		if (threadIdx.x == 0) //use one thread to count the carry for globe the warp
		{
			for (int i = 1; i < blockDim.x / 32; ++i)
			{
				smem[i] += smem[i - 1];
			}
		}
	__syncthreads();
	if (warp > 0)
		*index += smem[warp - 1];
	bool flag = false;
	if (blockDim.x % 32 == 0) flag = true;
	*carry = smem[blockDim.x / 32 - flag];
}

__global__ void convert2uintV2(float* input, unsigned int *output, int size, int threadNum)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int step = threadNum;
	for (int i = idx; i < size; i += threadNum)
	{
		output[i] = convert(input[i]);
	}
}



__device__ void collect_number(unsigned int *data, int limit, unsigned int pattern, unsigned int *ans, int *ansIndex, int *ansSize)
{
	__shared__ int smem[32]; //for count each warp's tmp carry
	*ansSize = 0;
	const int tid = threadIdx.x;
	int cot = 0;
	int carry;
	int index;
	int alibn_limit = limit;
	if (alibn_limit % blockDim.x) alibn_limit = alibn_limit + blockDim.x - (alibn_limit % blockDim.x);
	__syncthreads();
	for (int i = tid; i < alibn_limit; i += blockDim.x)
	{
		bool has_topk = false;
		if (i < limit&&data[i] > pattern)
		{
			has_topk = true;
		}
		gpu_check_warp(smem, has_topk, &carry, &index);
		if (carry>0)
		{
			if (has_topk)
			{
				ans[cot + index - 1] = data[i];
				ansIndex[cot + index - 1] = i;
			}
			cot += carry;
			//printf("%d %d %d %u\n",tid,carry,index,data[i]);
		}
		//************
		__syncthreads();
		/******/
	}
	if (threadIdx.x == 0)
	{
		*ansSize = cot;
	}
}


__device__ void collect_number_old(unsigned int *data, int n, int k,unsigned int pattern,unsigned int *ans, int *indexNum,int stride,int strideNum)
{
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int blockIndex = idy / stride;
	int offsetInBlock = idy % stride;
	int cot = 0;
	for (int i = stride * strideNum * blockIndex + offsetInBlock , j = 0; j < strideNum; j++, i += stride)
	{
		if (data[i] > pattern)
		{
			ans[cot] = data[i];
			indexNum[cot++] = j;
		}
	}
	/*if the cot < k ,so the left value must be desire*/
	if (cot < k)
	{
		for (int i = cot; i < k; ++i)
		{
			ans[i] = pattern;
		}
		//count the remain index and the data value must equal pattern
		for (int i = stride * strideNum * blockIndex + offsetInBlock, j = 0; j < strideNum; j++, i += stride)
		{
			if (data[i] == pattern)
			{
				indexNum[cot++] = j;
				if (cot == k) break;
			}
		}
	}
}

template<class T> __global__
//__global__ void radix_selection(unsigned int *data, int n, int k,unsigned int *ans,int *indexNum)
void KernelTopKRadixSelect(unsigned int * input, int stride, int strideNum, int blockNum, int k, T minValue, T * output, int * index,int limit)
{
	/* the idx th thread in one vector */
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	/* the idy th vector in one tensor */
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	int mask_desire = 0;
	unsigned int mask = 0x80000000;
	unsigned int desire = 0;
	__shared__ int pos_count[32*32];
	int tmp_k = k;
	//if (idx == 0)
		//printf("%d %d blockSize: <%d  ,%d>\n", idx + blockDim.x*idy,idy, blockDim.x, blockDim.y);
	int flag = 1;
	for (int i = 0; i < 32; i++)
	{
		pos_count[idx + blockDim.x*(idy%32)] = 0;
		if(flag)
		radixCount(input, stride*strideNum*blockNum, pos_count, mask, mask_desire, desire, stride, strideNum, blockNum);
		__syncthreads();
		int sumCount=0;
		for (int j = 0; j < 32; j++)
		{
			sumCount += pos_count[(idy % 32)*blockDim.x+j];
		}
		__syncthreads();
		if (tmp_k<sumCount)//this position should be 1
		{
			desire = mask^desire;
		}
		else //zoom out the k size,this position should be 0
		{
			tmp_k = tmp_k - sumCount;
			
			if (tmp_k == 0)
			{
				desire = (~(mask_desire >> 1)) | desire;

				// avoid Synchronize deadlock 
				//break;

				flag = 0;
			}
		}
		mask_desire = mask^mask_desire;
		mask = mask >> 1;
	}

	//************just for test**********************
	/*if (idy == 0 && idx == 0)
	{
		for (int i = 0; i < limit; i+=7)
		{
			for (int j = i; j<i + 8; j += 2)
				printf("num: %d  data: %u ", j, input[j]);
			printf("\n");
			i++;
			for(int j=i;j<i+8;j+=2)
				printf("num: %d  data: %u ", j, input[j]);
			printf("\n");
		}
	}*/
	__syncthreads();
	//***********************************************
	
	if (idx == 0)
	{
		unsigned int* uintOutput = new unsigned int;
		int* tmpIndex = new int;
		//*******************something worng***************************
		cudaMalloc((void **)&uintOutput, sizeof(unsigned int)* k);
		cudaMalloc((void **)&tmpIndex, sizeof(unsigned int)*k);
		//*************************************************************
		collect_number_old(input, limit,k, desire, uintOutput, tmpIndex,stride,strideNum);
		
		int blockIndex = idy / stride;
		int offsetInBlock = idy% stride;

		for (int i = stride * k * blockIndex + offsetInBlock , j = 0; j < k; j++, i += stride)
		{
			output[i] = deconvert(uintOutput[j]);
			index[i] = tmpIndex[j];
		}
	}
	__syncthreads();
}
template<class T> __global__ 
void KernelTopK(T * input, int stride, int strideNum, int blockNum, int k, T minValue, T * output, int * index)
{
    __shared__ CudaHeapNode<T> heapData[(SHARED_MEMORY_SIZE)/sizeof(CudaHeapNode<T>)];
	//printf("stride: %d, strideNum: %d, blockNum: %d,blockDim.x:%d,blockDim.y:%d\n", stride, strideNum, blockNum,blockDim.x,blockDim.y);
    /* worker index */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* index of the data arry along the given dimension */
    int j = blockDim.y * blockIdx.y + threadIdx.y;
//	printf("%d,%d,%d", blockDim.y, blockIdx.y, threadIdx.y);

    if (i >= strideNum || i >= blockDim.x || j >= stride * blockNum)
        return;

	//如果不需要该线程

    int blockIndex = j / stride; // j
    int offsetInBlock = j % stride; // 0
    T * d = input + stride * strideNum * blockIndex + offsetInBlock;
	//printf("j:%d,  position:%d\n", j, stride * strideNum * blockIndex + offsetInBlock);//200*30*blockIndex
    CudaXHeap<MIN_HEAP, T> heap(k, heapData + k * (threadIdx.y * blockDim.x + threadIdx.x));
    __syncthreads();

    /* go over the data array and build the heap */
    int indexOffset = blockDim.x;
    int dataOffset = stride * blockDim.x;

    if (i + (heap.size - 1) * indexOffset < strideNum) {
        int p = i;
        int q = i * stride;
        for (int m = 0; m < heap.size; m++) {
            heap.Push(p, d[q]);
            p += indexOffset;
            q += dataOffset;
        }

        for (; p < strideNum; p += indexOffset, q += dataOffset) {
            T v = d[q];
            if (v > heap.topValue){
                heap.ReplaceTop(p, v);
            }
        }
    }
    else {
        for(int p = i, q = i * stride; p < strideNum; p += indexOffset, q += dataOffset) {
            heap.Push(p, d[q]);
        }
    }

    /* fill the heap if no enough items are processed */
    while(heap.count < heap.size){
        heap.Push(-1, minValue);
    }

    __syncthreads();

    if(threadIdx.x == 0){
        CudaXHeap<MIN_HEAP, T> heapFinal(k, k, heapData + k * threadIdx.y * blockDim.x);

        /* merge the result over the workers. 
           This can be improved by parallel merging */
        if(blockDim.x > 1){
            for(int p = 1; p < blockDim.x && p < strideNum; p++){
                CudaHeapNode<T> * hd = heapData + k * (threadIdx.y * blockDim.x + p);
                for(int q = 0; q < k; q++){
                    if (hd[q].value > heapFinal.topValue)
                        heapFinal.ReplaceTop(hd[q]);
                }
            }
        }

        int offset = stride * k * blockIndex + offsetInBlock;
        T * dOutput = output + offset;
        int * indexOutput  = index + offset;

        /* pop for the final result */
        for(int q = k - 1; q >= 0; q--){
            dOutput[stride * q] = heapFinal.items[0].value;
            indexOutput[stride * q] = heapFinal.items[0].index;
            heapFinal.items[0] = heapFinal.items[heapFinal.count - 1];
            heapFinal.count--;
            heapFinal.Down(0);
        }
    }
}
/* 
get the top-k items along a given dimension 
>> a - input tensor
>> b - output tensor (top-k result)
>> index - index of the top-k items
>> dim - the dimension along which the sorting is performed 
>> k - how many items returned after sorting
*/
void CudaTopKRadixSelect(XTensor * a, XTensor * b, XTensor * index, int dim, int k)
{
	CheckNiuTransErrors((a->unitSize == b->unitSize), "Unmatched input tensors!");
	CheckNiuTransErrors((a->order == b->order), "Unmatched input tensors!");
	CheckNiuTransErrors((index == NULL || a->order == index->order), "Unmatched input tensors!");
	CheckNiuTransErrors((index->dataType == X_INT), "Wrong data type!");
	CheckNiuTransErrors((b->dimSize[dim] == k), "A too large K");
	// Tensor(100, 200, 300, 400, 500, 600)  dim = 2
	// thread(100x200x400x500x600, 300) = stride * blockNum
	int stride = 1;
	int strideNumA = a->dimSize[dim];
	for (int i = 0; i < dim; i++)
		stride *= a->dimSize[i];
	// stride = 100 * 200
	int blockNum = 1;
	for (int i = dim + 1; i < a->order; i++)
		blockNum *= a->dimSize[i];
	//blockNum = 400 * 500 * 600
	int workerNum = 32; // should be tuned for better performance
	int cudaGrids[3];
	int cudaBlocks[3];
	GDevs->GetGridAndBlockSize2D(a->mem->devID,
		workerNum, stride * blockNum, MAX_INT,
		cudaGrids, cudaBlocks);

	printf("\n");
	printf("block size :%d, %d\n", cudaBlocks[0],cudaBlocks[1]);
	printf("grid size :%d, %d", cudaGrids[0], cudaGrids[1]);
	printf("\n");
}
void CudaTopK(XTensor * a, XTensor * b, XTensor * index, int dim, int k)
{
    CheckNiuTransErrors((a->unitSize == b->unitSize), "Unmatched input tensors!");
    CheckNiuTransErrors((a->order == b->order), "Unmatched input tensors!");
    CheckNiuTransErrors((index == NULL || a->order == index->order), "Unmatched input tensors!");
    CheckNiuTransErrors((index->dataType == X_INT), "Wrong data type!");
    CheckNiuTransErrors((b->dimSize[dim] == k), "A too large K");
	CudaTopKRadixSelect(a, b, index, dim, k);
	// Tensor(100, 200, 300, 400, 500, 600)  dim = 2
	// thread(100x200x400x500x600, 300)
    int stride = 1;
    int strideNumA = a->dimSize[dim];
    for(int i = 0; i < dim; i++)
        stride *= a->dimSize[i];

    int blockNum = 1;
    for(int i = dim + 1; i < a->order; i++)
        blockNum *= a->dimSize[i];
    int workerNum = 32; // should be tuned for better performance

    int cudaGrids[3];
    int cudaBlocks[3];

    GDevs->GetGridAndBlockSize2D(a->mem->devID,
        workerNum, stride * blockNum, MAX_INT,
        cudaGrids, cudaBlocks);

    for (int i = 0; i < 2; i++) {
        if ((cudaBlocks[0] * cudaBlocks[1] + 1) * k * (a->unitSize + sizeof(int)) >= SHARED_MEMORY_SIZE) {
            if (cudaBlocks[1] >= 2) {
                cudaBlocks[1] /= 2;
                cudaGrids[1] *= 2;
            }
        }

        if ((cudaBlocks[0] * cudaBlocks[1] + 1) * k * (a->unitSize + sizeof(int)) >= SHARED_MEMORY_SIZE) {
            if (cudaBlocks[0] >= 2) {
                cudaBlocks[0] /= 2;
                cudaGrids[0] *= 2;
            }
        }
    }
    /* we run the kernel if the heaps can fit into the shared memory */
    if ((cudaBlocks[0] * cudaBlocks[1] + 1) * k * (a->unitSize + sizeof(int)) < SHARED_MEMORY_SIZE) {
        if (a->dataType == DTYPE_IN_MATRIX){
            KernelTopK<DTYPE> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                ((DTYPE*)a->data, stride, strideNumA, blockNum, k, DTYPE_MIN,
                                 (DTYPE*)b->data, (int*)index->data);

			float* input  = (float*)malloc(sizeof(float)*strideNumA*blockNum);
			cudaMemcpy(input,a->data,sizeof(float)*strideNumA*blockNum,cudaMemcpyDeviceToHost);
			/*
			for (int i = 0; i < a->order; i++)
			{
				printf("dim=%d  %d\n",i, a->dimSize[i]);
			}
			printf("strideNumA  %d\n", strideNumA);
			printf("blockNum  %d\n", blockNum);
			printf("dimSize  %d\n", a->dimSize[dim]);
			printf("dim  %d\n", dim);
			*/
			//for (int i = 0; i < strideNumA*blockNum*a->dimSize[dim]; i++)
				//printf("%f\n",input[i]);
			
			unsigned int* output = (unsigned int*)malloc(sizeof(unsigned int)*strideNumA*blockNum*stride), *goutput;
			cudaMalloc(&goutput, sizeof(unsigned int)*strideNumA*blockNum*stride);
			convert2uintV2 << <1, 128 >> >((float*)a->data, goutput, strideNumA*blockNum*stride, 128);
			cudaMemcpy(output, goutput, stride*strideNumA*blockNum * sizeof(float), cudaMemcpyDeviceToHost);
			KernelTopKRadixSelect<DTYPE> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
				(goutput, stride, strideNumA, blockNum, k, DTYPE_MIN,
				(DTYPE*)b->data, (int*)index->data, stride*strideNumA*blockNum);

			int *indexTensorData = (int *)malloc(4 * strideNumA*blockNum*stride);
			cudaMemcpy(indexTensorData, index->data, sizeof(DTYPE)*index->unitNum, cudaMemcpyDeviceToHost);
			for (int i = 0; i <160; ++i)
			{
				printf("%d ", indexTensorData[i]);
			}
			printf("__________________finish op___________________\n");
        }
        else {
            ShowNiuTransErrors("TODO!");
        }

    }
    /* we resort to sorting if the data cannot fit inside the shared memory */
    else {
        int dimSize[MAX_TENSOR_DIM_NUM];
        memcpy(dimSize, a->dimSize, sizeof(int) * a->order);
        dimSize[0] = -dimSize[0];
        XTensor * indexA = new XTensor(a->order, dimSize, 1.0F, X_INT, a->mem);
        indexA->data = a->mem->AllocBuf(a->devID, a->unitNum * sizeof(int));

        /* make the index tensor */
        indexA->SetAscendingOrder(dim);

        CudaSortBig(a, b, indexA, index, dim, k);

        a->mem->ReleaseBuf(a->devID, a->unitNum * sizeof(int));
        delete indexA;
    }
}

/* 
summation of data arrays (CUDA Kernel) 
c = a  + b * \beta
>> a - A matrix
>> b - another matrix
>> c - where we put a+b
>> size - the size of a/b/c
>> beta - the coefficient
*/
extern "C" __global__ 
void KernelADD(DTYPE * a, DTYPE * b, DTYPE * c, int size, DTYPE beta)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        c[i] = a[i] + b[i] * beta;
}

/* 
tensor summation c = a + b * \beta (cuda version) 
>> a - a tensor
>> b - another tensor
>> c - where we put a+b*\beta. we save it in a if c is NULL
>> beta - the scaling factor
*/
void CudaSum(XTensor * a, XTensor * b, XTensor * c, DTYPE beta)
{
    if(c == NULL)
        c = a;

    CheckNiuTransErrors((a->unitNum == b->unitNum && a->unitNum == c->unitNum), 
                        "Unmatched tensors in addition!");

    CheckNiuTransErrors((a->dataType == b->dataType && a->dataType == c->dataType), 
                        "Unmatched tensors in addition!");

#if !defined(CUDA_UVA)
    CheckNiuTransErrors(a->devID == b->devID && a->devID == c->devID, 
                        "Matrices used in summation are not on the same GPU.");
#endif

    if(!a->isSparse && !b->isSparse){
        CheckNiuTransErrors(!c->isSparse, 
                            "Illegal use of sparse matrix in addition!");

        if(a->dataType == DTYPE_IN_MATRIX && 
           b->dataType == DTYPE_IN_MATRIX && 
           c->dataType == DTYPE_IN_MATRIX)
        {
            cublasHandle_t * handleA = a->mem->GetCublasHandle();
            cublasHandle_t * handleB = a->mem->GetCublasHandle();
            cublasHandle_t * handle = *handleA != 0 ? handleA : handleB;

            if(c == a && *handle != 0){
#ifdef DOUBELPRICSION
                cublasDaxpy(*handle, a->unitNum, &beta, (DTYPE*)b->data, 1, (DTYPE*)a->data, 1);
#else
                cublasSaxpy(*handle, a->unitNum, &beta, (DTYPE*)b->data, 1, (DTYPE*)a->data, 1);
#endif
            }
            else{
                int gridSize[3], blockSize[3];

                GDevs->GetGridAndBlockSize(a->devID, a->unitNum, gridSize, blockSize);

                dim3 blocks(gridSize[0]);
                dim3 threads(blockSize[0]);

                KernelADD<<<blocks, threads>>>((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, a->unitNum, beta);
            }
        }
        else{
            // TODO!!
            ShowNiuTransErrors("TODO!");
        }
    }
    else{
        // TODO!!
        ShowNiuTransErrors("TODO!");
    }
}

/* summation over arrays 
tensor summation c = a + b * \beta (cuda version) with an input handle
>> devID - device ID (MUST >= 0)
>> handle - cuda handle
>> a - an array
>> b - another array
>> c - where we put a+b
>> size - size of the array
>> beta - the coefficient
*/
void CudaSumWithHandle(int devID, cublasHandle_t * handle, DTYPE * a, DTYPE * b, DTYPE * c, int size, DTYPE beta)
{
    if(size == 0)
        return;

    if(c == NULL)
        c = a;

    CheckNiuTransErrors((a && b && c), "Empty arrays in addition!");

    if(c == a){
#ifdef DOUBELPRICSION
        cublasDaxpy(*handle, size, &beta, b, 1, a, 1);
#else
        cublasSaxpy(*handle, size, &beta, b, 1, a, 1);
#endif
    }
    else{
        int gridSize[3], blockSize[3];

        GDevs->GetGridAndBlockSize(devID, size, gridSize, blockSize);

        dim3 blocks(gridSize[0]);
        dim3 threads(blockSize[0]);

        KernelADD<<<blocks, threads>>>((DTYPE*)a, (DTYPE*)b, (DTYPE*)c, size, beta);
    }
}

/* 
summation of a tensor and a vector (column vector) 
c_col = a_col  + b * \beta
>> a - a tensor
>> b - a vector with the same column size with a
>> c - where we put a+b. we save it in a
>> colNum - column number (of a block)
>> blockSize - size of a block
>> size - size of the entire data array
>> beta - the scaling factor
*/
extern "C" __global__ 
void KernelADDByColumnTV(DTYPE * a, DTYPE * b, DTYPE * c, int colNum, int blockSize, int size, DTYPE beta)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= size)
        return;

    int offset = i % blockSize;
    int row = offset / colNum;

    c[i] = a[i] + b[row] * beta;
}

/* 
summation of a tensor and a vector (column vector) 
for each column a_col (in a block), we have
c_col = a_col + b * \beta
where b is a vector.

>> a - a tensor
>> b - a vector with the same column size with a
>> c - where we put a+b. we save it in a if c is NULL
>> beta - the scaling factor
*/
void CudaSumByColumnTV(XTensor * a, XTensor * b, XTensor * c, DTYPE beta)
{
    if(c == NULL)
        c = a;

    CheckNiuTransErrors((a && b && c), "Empty input tensors!");
    CheckNiuTransErrors((XTensor::IsIdentical(a, c)), "Unmatched tensors in addition!");
    CheckNiuTransErrors((b->order == 2 && b->dimSize[0] == 1 && b->dimSize[1] == a->dimSize[1]),
                        "Illegal input vector size!");
    CheckNiuTransErrors((a->dataType == DTYPE_IN_MATRIX && b->dataType == DTYPE_IN_MATRIX &&
                         c->dataType == DTYPE_IN_MATRIX), "TODO");

    int colNum = a->dimSize[0];
    int rowNum = a->dimSize[1];
    int blockNum = 1;
    for(int i = 2; i < a->order; i++)
        blockNum *= a->dimSize[i];

    int cudaGridSize[3];
    int cudaBlockSize[3];

    GDevs->GetGridAndBlockSize(c->devID, a->unitNum, cudaGridSize, cudaBlockSize);

    KernelADDByColumnTV <<<dim3(cudaGridSize[0]), dim3(cudaBlockSize[0]) >>>
                         ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, colNum, rowNum * colNum, a->unitNum, beta);
}

/* 
summation of a vector (column vector) and a tensor
c = a + \sum{col} b_col * \beta
>> a - a vector with the same column size with b
>> b - a tensor 
>> c - where we put a+b. we save it in a
>> colNum - column number (of a block)
>> blockSize - size of a block
>> size - size of the entire data array
>> beta - the scaling factor
*/
extern "C" __global__ 
void KernelADDByColumnVT(DTYPE * a, DTYPE * b, DTYPE * c, int colNum, int rowNum, int blockNum, DTYPE beta)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= rowNum)
        return;

    DTYPE sum = 0;
    for(int k = 0; k < blockNum; k++){
        DTYPE * bp = b + (rowNum * k + row) * colNum;
        if(colNum % 4 == 0){
            for(int i = 0; i < colNum; i += 4)
                sum += bp[i] + bp[i + 1] + b[i + 2] + b[i + 3];
        }
        else if(colNum % 2 == 0){
            for(int i = 0; i < colNum; i += 2)
                sum += bp[i] + bp[i + 1];
        }
        else{
            for(int i = 0; i < colNum; i++)
                sum += bp[i];
        }
        __syncthreads();
    }

    c[row] = a[row] + beta * sum;
}

/* 
summation of a vector (column vector) and a tensor

for each column b_col, we have
c = a + \sum{col} b_col * \beta
where c and a are vectors, and b_col is a column in b.

>> a - a vector with the same column size with b
>> b - a tensor 
>> c - where we put a+b. we save it in a if c is NULL
>> beta - the scaling factor
*/
void CudaSumByColumnVT(XTensor * a, XTensor * b, XTensor * c, DTYPE beta)
{
    if(c == NULL)
        c = a;

    CheckNiuTransErrors((a && b && c), "Empty input tensors!");
    CheckNiuTransErrors((XTensor::IsIdentical(a, c)), "Unmatched tensors in addition!");
    CheckNiuTransErrors((a->order == 2 && a->dimSize[0] == 1 && b->dimSize[1] == a->dimSize[1]),
                        "Illegal input vector size!");
    CheckNiuTransErrors((a->dataType == DTYPE_IN_MATRIX && b->dataType == DTYPE_IN_MATRIX &&
                         c->dataType == DTYPE_IN_MATRIX), "TODO");

    int colNum = b->dimSize[0];
    int rowNum = b->dimSize[1];
    int blockNum = 1;
    for(int i = 2; i < b->order; i++)
        blockNum *= b->dimSize[i];

    int cudaGridSize[3];
    int cudaBlockSize[3];

    GDevs->GetGridAndBlockSize(c->devID, a->dimSize[1], cudaGridSize, cudaBlockSize);

    KernelADDByColumnVT <<<dim3(cudaGridSize[0]), dim3(cudaBlockSize[0]) >>>
                         ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, colNum, rowNum, blockNum, beta);
}

/* 
mutilication of a dense matrix with a sparse matrix 
c = a * b * \alpha

>> a - a dense matrix
>> transposedA - indicates whether a is transposed
>> aColSize - column size of matrix a
>> aRowSize - row size of matrix a
>> b - a sparse matrix
>> transposedA - indicates whether b is transposed
>> bNonZeroNum - number of non-zero items in b
>> bColSize - column size of matrix b
>> bRowSize - row size of matrix b
>> c - the resulting (dense) matrix
>> alpha - the scaling factor
*/
extern "C" __global__
 void KernelMatrixMulDenseMSparseMV2(DTYPE * a, MATRIX_TRANS_TYPE transposedA, int aColSize, int aRowSize, 
                                     void * b, MATRIX_TRANS_TYPE transposedB, int bNonZeroNum, int bColSize, int bRowSize, 
                                     DTYPE * c, int cColSize, int cRowSize, DTYPE alpha)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    char * bData = (char*)b;
    int tupleSize = sizeof(int) + sizeof(DTYPE);

    for(int k = 0; k < bNonZeroNum; k += blockDim.x){
        __shared__ int   bEntryRow[MAX_CUDA_THREAD_NUM_PER_BLOCK];
        __shared__ int   bEntryCol[MAX_CUDA_THREAD_NUM_PER_BLOCK];
        __shared__ DTYPE bValue[MAX_CUDA_THREAD_NUM_PER_BLOCK];

        if(k + threadIdx.x < bNonZeroNum){
            /* load the sub-block of the sparse matrix b */
            int key = *(int*)(bData + tupleSize * (k + threadIdx.x));

            bEntryRow[threadIdx.x] = key / bRowSize;
            bEntryCol[threadIdx.x] = key % bRowSize;
            bValue[threadIdx.x] = *(DTYPE*)(bData + tupleSize * (k + threadIdx.x) + sizeof(int));
        }

        /* synchronize to make sure the sub-block of the sparse matrix b is loaded */
        __syncthreads();

        if(i < cColSize){
            if(transposedA == X_NOTRANS && transposedB == X_NOTRANS){
                for(int m = 0; m < blockDim.x && k + m < bNonZeroNum; m++){
                    DTYPE * aRow = a + aRowSize * i;
                    c[i * cRowSize + bEntryCol[m]] += aRow[bEntryRow[m]] * bValue[m] * alpha;
                }
            }
            else if(transposedA == X_TRANS && transposedB == X_NOTRANS){
                for(int m = 0; m < blockDim.x && k + m < bNonZeroNum; m++){
                    DTYPE * aCol = a + i;
                    c[i * cRowSize + bEntryCol[m]] += aCol[bEntryRow[m] * aRowSize] * bValue[m] * alpha;
                }
            }
            else if(transposedA == X_NOTRANS && transposedB == X_TRANS){
                for(int m = 0; m < blockDim.x && k + m < bNonZeroNum; m++){
                    DTYPE * aRow = a + aRowSize * i;
                    c[i * cRowSize + bEntryRow[m]] += aRow[bEntryCol[m]] * bValue[m] * alpha;
                }
            }
            else if(transposedA == X_TRANS && transposedB == X_TRANS){
                for(int m = 0; m < blockDim.x && k + m < bNonZeroNum; m++){
                    DTYPE * aCol = a + i;
                    c[i * cRowSize + bEntryRow[m]] += aCol[bEntryCol[m] * aRowSize] * bValue[m] * alpha;
                }
            }
        }

        /* synchronize to the preceding computation is done before loading new sub-blocks */
        __syncthreads();
    }
}


/* 
matrix multiplication (for 2d tensors) (cuda version)
c = trans(a) * trans(b) * alpha + c * beta 
where trans() return the transposed matrix if the flag is fired
>> a - tensor a
>> transposedA - indicates whether the matrices in a are transposed
>> b - tensor b
>> transposedB - indicates whether teh matrices in b are transposed
>> c - where we put a*b
>> alpha - a coefficient
>> beta - another coefficient
>> stream - the string for creating the job pipeline
*/
void CudaMatrixMul2D(XTensor * a, MATRIX_TRANS_TYPE transposedA, 
                     XTensor * b, MATRIX_TRANS_TYPE transposedB, 
                     XTensor * c, 
                     DTYPE alpha, DTYPE beta, XStream * stream)
{
    int an = transposedA == X_TRANS ? a->dimSize[0] : a->dimSize[1];
    int am = transposedA == X_TRANS ? a->dimSize[1] : a->dimSize[0];
    int bn = transposedB == X_TRANS ? b->dimSize[0] : b->dimSize[1];
    int bm = transposedB == X_TRANS ? b->dimSize[1] : b->dimSize[0];
    int cn = c->dimSize[1];
    int cm = c->dimSize[0];
        
    CheckNiuTransErrors((a && b && c), 
                       "Empty matrices in multiplication!");

    CheckNiuTransErrors((am == bn && an == cn && bm == cm), 
                       "Unmatched matrices in multiplication!");

    CheckNiuTransErrors((a->devID >= 0), "Cuda version matrix mutiplication must be run on GPUs.");

    CheckNiuTransErrors(a->devID == b->devID && a->devID == c->devID, 
                        "Matrices used in multiplication are not on the same GPU.");

    /* a dense matrix multiply a dense matrix */
    if(!a->isSparse && !b->isSparse){
        CheckNiuTransErrors((!c->isSparse), "Illegal use of sparse matrix in multiplication!");

        //cublasHandle_t * handle = GDevs->GetCudaHandle(a->devID);
        cublasHandle_t * handle = a->mem->GetCublasHandle();

        /* !!!! might have problems */
        if(stream != NULL)
            cublasSetStream(*handle, stream->stream);

        if(a->dataType == X_FLOAT && b->dataType == X_FLOAT && c->dataType == X_FLOAT){
            CudaBLASMatrixMUL(handle, a->data, transposedA, a->dataType, b->data, transposedB, a->dataType, c->data, c->dataType,
                              a->dimSize[1], a->dimSize[0], b->dimSize[1], b->dimSize[0], c->dimSize[1], c->dimSize[0], 
                              alpha, beta);
        }
        else{
            // TODO!!
            ShowNiuTransErrors("TODO!");
        }
    }
    /* a dense matrix multiply a sparse matrix */
    else if(!a->isSparse && b->isSparse){

        CheckNiuTransErrors(!c->isSparse, "Illegal use of sparse matrix in multiplication!");
        CheckNiuTransErrors((beta == 0 || beta == 1.0), "beta must be 0 or 1.");

        if(a->dataType == DTYPE_IN_MATRIX && b->dataType == DTYPE_IN_MATRIX && c->dataType == DTYPE_IN_MATRIX){
            int gridSize[3], blockSize[3];

            GDevs->GetGridAndBlockSize(c->devID, a->dimSize[1], gridSize, blockSize);

            dim3 blocks(gridSize[0]);
            dim3 threads(blockSize[0]);

            void * bData = (void*)((char*)b->data + sizeof(int));

            if(beta == 0)
                c->SetZeroAll();
            else if(beta != 1.0F)
                XTensor::ScaleAndShift(c, beta, 0);

            KernelMatrixMulDenseMSparseMV2<<<blocks, threads>>>((DTYPE*)a->data, transposedA, a->dimSize[1], a->dimSize[0], 
                                                                 bData, transposedB, b->unitNumNonZero, b->dimSize[1], b->dimSize[0],
                                                                (DTYPE*)c->data, c->dimSize[1], c->dimSize[0], alpha);
        }
        else{
            // TODO!!
            ShowNiuTransErrors("TODO!");
        }

    }
    else{
        // TODO!!
        ShowNiuTransErrors("TODO!");
    }
}

/* 
multiplication of data arrays in a element-wise manner c(i) = a(i)*b(i) 
>> a - data array a
>> b - data array b
>> c - result data array
>> size - size of c
*/
extern "C" __global__ 
void KernelMulElementWise(DTYPE * a, DTYPE * b, DTYPE * c, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        c[i] = a[i] * b[i];
}

/*
multiplication of data arrays in a element-wise manner c(i) = a(i)*b(i) + \alpha*c(i) 
>> a - data array a
>> b - data array b
>> c - result data array
>> size - size of c
>> alpha - the coefficient
*/
extern "C" __global__ 
void KernelMulElementWiseV2(DTYPE * a, DTYPE * b, DTYPE * c, int size, DTYPE alpha)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        c[i] = a[i] * b[i] + alpha * c[i];
}

/* 
multiplication of two tensors in a element-wise manner c(i) = a(i)*b(i). 
Note that a and b can be of different sizes here, i.e., 
|a_lead| <= |c_lead| and |b_lead| <= |c_lead|
where |a_lead| means the size of the leading dimension of a
>> a - tensor a
>> b - tensor b
>> c - result tensor
>> stride - the number of items we go over when move next along the leading dimension in a block
>> ldSizeA - size of the leading dimension of a
>> ldSizeB - size of the leading dimension of b
>> ldSizeC - size of the leading dimension of c
>> blockNum - number of blocks
*/
template<int nonZeroAlpha> __global__ 
void KernelMulElementWiseTensorDynamic(DTYPE * a, DTYPE * b, DTYPE * c, DTYPE alpha, 
                                       int stride, int ldSizeA, int ldSizeB, int ldSizeC, int blockNum)
{
    __shared__ DTYPE* ap[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE* bp[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE* cp[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= blockNum * stride || j >= ldSizeC)
        return;

    if(threadIdx.y == 0){
        int block = i / stride;
        int size  = block * stride;
        ap[threadIdx.x] = a + size * ldSizeA;
        bp[threadIdx.x] = b + size * ldSizeB;
        cp[threadIdx.x] = c + size * ldSizeC;
    }

    __syncthreads();

    int aj= j >= ldSizeA ? j % ldSizeA : j;
    int bj= j >= ldSizeB ? j % ldSizeB : j;
    int offseti = i % stride;

    if(nonZeroAlpha == 0)
        cp[threadIdx.x][j * ldSizeC + offseti] = ap[threadIdx.x][aj* ldSizeA + offseti] * bp[threadIdx.x][bj* ldSizeB + offseti];
    else
        cp[threadIdx.x][j * ldSizeC + offseti] = ap[threadIdx.x][aj* ldSizeA + offseti] * bp[threadIdx.x][bj* ldSizeB + offseti] +
                                                 alpha * cp[threadIdx.x][j * ldSizeC + offseti];
}

/* 
element-wise product of two tensors 
c(i) = a(i)*b(i) + \alpha * c(i)
where i is the item index
>> a - tensor a
>> b - tensor b
>> c - result tensor
>> leadingDim - leading dimension
>> alpha - the coefficient
*/
extern "C" 
void CudaMultiplyElementWise(XTensor * a, XTensor * b, XTensor * c, int leadingDim, DTYPE alpha)
{
    CheckNiuTransErrors((a->unitNum <= c->unitNum && b->unitNum <= c->unitNum), 
                        "Unmatched tensors in multiplication!");
    CheckNiuTransErrors((a->order == b->order && a->order == c->order), "Unmatched tensors!");

    int stride = 1;
    int blockSizeA = 1;
    int blockNum = 1;
    int dimensionSizeA = a->dimSize[leadingDim];
    int dimensionSizeB = b->dimSize[leadingDim];
    int dimensionSizeC = c->dimSize[leadingDim];

    for(int i = 0; i < a->order; i++){
        if(i != leadingDim){
            CheckNiuTransErrors((a->dimSize[i] == b->dimSize[i] &&
                                 a->dimSize[i] == c->dimSize[i]),
                                "Unmatched tensors!");
        }
        if(i < leadingDim)
            stride *= a->dimSize[i];
    }
    
    blockSizeA = stride * dimensionSizeA;
    blockNum = a->unitNum / blockSizeA;

    if(!a->isSparse && !b->isSparse){
        if(a->dataType == DTYPE_IN_MATRIX && b->dataType == DTYPE_IN_MATRIX){
            int cudaGridSize[3];
            int cudaBlockSize[3];

            if(a->unitNum == c->unitNum && b->unitNum == c->unitNum){
                GDevs->GetGridAndBlockSize(a->devID, c->unitNum, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[0]), threads(cudaBlockSize[0]);

                if(alpha == 0)
                    KernelMulElementWise<<<blocks, threads>>>((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, c->unitNum);
                else
                    KernelMulElementWiseV2<<<blocks, threads>>>((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, c->unitNum, alpha);
            }
            else{
                GDevs->GetGridAndBlockSize2D(c->devID, stride * blockNum, dimensionSizeC, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[0], cudaGridSize[1]), threads(cudaBlockSize[0], cudaBlockSize[1]);

                if(alpha == 0){
                    KernelMulElementWiseTensorDynamic<0> <<<blocks, threads>>>
                                                         ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, 0,
                                                           stride, dimensionSizeA, dimensionSizeB, dimensionSizeC, 
                                                           blockNum);
                }
                else{
                    KernelMulElementWiseTensorDynamic<1> <<<blocks, threads>>>
                                                         ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, alpha,
                                                           stride, dimensionSizeA, dimensionSizeB, dimensionSizeC, 
                                                           blockNum);
                }
            }
        }
        else{
            // TODO!!
            ShowNiuTransErrors("TODO!");
        }
    }
    else{
        // TODO!!
        ShowNiuTransErrors("TODO!");
    }
}

/* 
reduce a tensor to another that keeps the sum along a dimension  - slow version

Given a block of data, we go over each dimension i in the stride and we have
sum_i = sum_{0<=j<strideNum} exp(input_{i,j} - shift) if isExp == true;
      = sum_{0<=j<strideNum} input_{i,j} - shift if isExp == false;
where we can view the block as a matrix and input_{i,j} represent the item at the
crossing of the i-th columne and the j-th row.

>> input - the input array (representing a tensor)
>> output - the sum over each block. NOTE: output is also an array
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to finish the reduce
>> reducedStrideNum - the number of strides after reducation 
>> blockSize - size of the block (i.e., stride * strideNum)
>> blockNum - how many blocks
>> shift - the bias imposed on the input
>> power - power of the item in the array
>> isExp - specify if we perform exp() on the input
*/
 __global__
void KernelReduceSum(DTYPE * input, DTYPE * output, 
                     int stride, int strideNum, int reducedStrideNum, 
                     int blockSize, int blockNum, 
                     DTYPE * shift, DTYPE power, bool isExp)
{
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK * MIN_CUDA_SHARED_MEM_COL_SIZE/2];
    __shared__ DTYPE bias[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    int idx = threadIdx.x * blockDim.y + threadIdx.y;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    if(i >= stride * blockNum)
        return;

    if(threadIdx.y == 0)
        bias[threadIdx.x] = shift != NULL ? shift[i] : 0;

    __syncthreads();

    int k = i / stride;
    int iOffset = i % stride;
    bool isValid = (i < stride * blockNum && j < strideNum);

    DTYPE value =  isValid ? input[blockSize * k + stride * j + iOffset] - bias[threadIdx.x] : 0;

    if(power != (DTYPE)1.0){
        if(power == (DTYPE)2.0)
            value = value * value;
        else if(power == (DTYPE)0.5)
            value = sqrt(value);
        else
            value = pow(value, power);
    }

    if(isExp && isValid)
        value = exp(value);

    /* load data into the shared mem */
    iData[threadIdx.x * blockDim.y + threadIdx.y] = value;

    __syncthreads();

    /* do reduction in shared mem */
    for (unsigned int s = blockDim.y/2; s > 0; s >>= 1){
        if (threadIdx.y < s)
            iData[idx] += iData[idx + s];

        __syncthreads();
    }

    /* write result for this block to the output array */
    if (threadIdx.y == 0 && blockIdx.y < reducedStrideNum) 
        output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = iData[threadIdx.x * blockDim.y];
}

 /* 
reduce a tensor to another that keeps the sum along a dimension  - slow version
This is for float16 reduction.

Given a block of data, we go over each dimension i in the stride and we have
sum_i = sum_{0<=j<strideNum} exp(input_{i,j} - shift) if isExp == true;
      = sum_{0<=j<strideNum} input_{i,j} - shift if isExp == false;
where we can view the block as a matrix and input_{i,j} represent the item at the
crossing of the i-th columne and the j-th row.

>> input - the input array (representing a tensor)
>> output - the sum over each block. NOTE: output is also an array
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to finish the reduce
>> reducedStrideNum - the number of strides after reducation 
>> blockSize - size of the block (i.e., stride * strideNum)
>> blockNum - how many blocks
>> shift - the bias imposed on the input
>> power - power of the item in the array
>> isExp - specify if we perform exp() on the input
*/
 __global__
void KernelReduceSum(__half * input, __half * output, 
                     int stride, int strideNum, int reducedStrideNum, 
                     int blockSize, int blockNum, 
                     __half * shift, __half power, bool isExp)
{
    int idx = threadIdx.x * blockDim.y + threadIdx.y;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    if(i >= stride * blockNum)
        return;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    __shared__ __half iData[MAX_CUDA_THREAD_NUM_PER_BLOCK * MIN_CUDA_SHARED_MEM_COL_SIZE/2];
    __shared__ __half bias[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    if(threadIdx.y == 0)
        bias[threadIdx.x] = shift != NULL ? shift[i] : __half(0);
#else
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK * MIN_CUDA_SHARED_MEM_COL_SIZE/2];
    __shared__ DTYPE bias[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    if(threadIdx.y == 0)
        bias[threadIdx.x] = shift != NULL ? __half(shift[i]) : __half(0);
#endif

    __syncthreads();

    int k = i / stride;
    int iOffset = i % stride;
    bool isValid = (i < stride * blockNum && j < strideNum);

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    __half value = isValid ? __hsub(input[blockSize * k + stride * j + iOffset], bias[threadIdx.x]) : __half(0);
    DTYPE power2 = __half2float(power);
    if(power2 != (DTYPE)1.0){
        if(power2 == (DTYPE)2.0)
            value = __hmul(value, value);
        else if(power2 == (DTYPE)0.5)
            value = hsqrt(value);
    }

    if(isExp && isValid)
        value = hexp(value);
#else
    DTYPE value =  isValid ? __half2float(input[blockSize * k + stride * j + iOffset]) - __half2float(bias[threadIdx.x]) : 0;
    DTYPE power2 = __half2float(power);

    if(power2 != (DTYPE)1.0){
        if(power2 == (DTYPE)2.0)
            value = value * value;
        else if(power2 == (DTYPE)0.5)
            value = sqrt(value);
        else
            value = pow(value, power2);
    }

    if(isExp && isValid)
        value = exp(value);
#endif

    /* load data into the shared mem */
    iData[threadIdx.x * blockDim.y + threadIdx.y] = value;

    __syncthreads();

    /* do reduction in shared mem */
    for (unsigned int s = blockDim.y/2; s > 0; s >>= 1){
        if (threadIdx.y < s)
            iData[idx] += iData[idx + s];

        __syncthreads();
    }

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    /* write result for this block to the output array */
    if (threadIdx.y == 0 && blockIdx.y < reducedStrideNum) 
        output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = iData[threadIdx.x * blockDim.y];
#else
    /* write result for this block to the output array */
    if (threadIdx.y == 0 && blockIdx.y < reducedStrideNum) 
        output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = __half(iData[threadIdx.x * blockDim.y]);
#endif

}

/* 
reduce a tensor to another that keeps the sum along a dimension  - fast version
>> input - the input array (representing a tensor)
>> output - the sum over each block. NOTE: output is also an array
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to finish the reduce
>> reducedStrideNum - the number of strides after reducation 
>> blockSize - size of the block (i.e., stride * strideNum)
>> blockNum - how many blocks
>> shift - the bias imposed on the input
>> power - power of the item in the array
>> isExp - specify if we perform exp() on the input
*/
template <unsigned int goodSize> __global__
void KernelReduceSumFast(DTYPE * input, DTYPE * output, 
                         int stride, int strideNum, int reducedStrideNum, 
                         int blockSize, int blockNum,
                         DTYPE * shift, DTYPE power, bool isExp)
{
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE bias[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    unsigned int tid = threadIdx.y;
    unsigned int j = blockIdx.y * (blockDim.y * 2) + threadIdx.y;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= stride * blockNum)
        return;
    
    if (threadIdx.y == 0)
        bias[threadIdx.x] = shift != NULL ? shift[i] : 0;

    __syncthreads();

    /* first level reduction */
    int k = i / stride;
    int iOffset = i % stride;

    bool isValid = j < strideNum;
    bool isValid2 = j + blockDim.y < strideNum;

    DTYPE * data =  iData + threadIdx.x * blockDim.y;
    DTYPE * inputData = input  + k * blockSize;
    DTYPE value  = isValid ? inputData[j * stride + iOffset] - bias[threadIdx.x]: 0;
    DTYPE value2 = isValid2 ? inputData[(j + blockDim.y) * stride + iOffset] - bias[threadIdx.x]: 0;
    
    if(power != (DTYPE)1.0){
        if(power == (DTYPE)2.0){
            value = value * value;
            value2 = value2 *value2;
        }
        else if(power == (DTYPE)0.5){
            value = sqrt(value);
            value2 = sqrt(value2);
        }
        else{
            value = pow(value, power);
            value2 = pow(value2, power);
        }
    }

    if(isExp){
        if(isValid)
            value = exp(value);
        if(isValid2)
            value2 = exp(value2);
    }

    /* load data into the shared mem */
    data[tid] = value + value2;

    __syncthreads();

    /* unroll the warp */
    if(goodSize >= 512) {if(tid < 256) {data[tid] += data[tid + 256];} __syncthreads();}
    if(goodSize >= 256) {if(tid < 128) {data[tid] += data[tid + 128];} __syncthreads();}
    if(goodSize >= 128) {if(tid <  64) {data[tid] += data[tid +  64];} __syncthreads();}
    if(goodSize >= 64)  {if(tid <  32) {data[tid] += data[tid +  32];} __syncthreads();}
    if(goodSize >= 32)  {if(tid <  16) {data[tid] += data[tid +  16];} __syncthreads();}
    if(goodSize >= 16)  {if(tid <   8) {data[tid] += data[tid +   8];} __syncthreads();}
    if(goodSize >=  8)  {if(tid <   4) {data[tid] += data[tid +   4];} __syncthreads();}
    if(goodSize >=  4)  {if(tid <   2) {data[tid] += data[tid +   2];} __syncthreads();}
    if(goodSize >=  2)  {if(tid <   1) {data[tid] += data[tid +   1];} __syncthreads();}

    /* write result for this block to the output array */
    if(threadIdx.y == 0 && blockIdx.y < reducedStrideNum) 
        output[(k * reducedStrideNum + blockIdx.y) * stride  + iOffset] = data[0];
}

/* 
reduce a tensor to another that keeps the sum along a dimension  - fast version
This is for float16 reduction

>> input - the input array (representing a tensor)
>> output - the sum over each block. NOTE: output is also an array
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to finish the reduce
>> reducedStrideNum - the number of strides after reducation 
>> blockSize - size of the block (i.e., stride * strideNum)
>> blockNum - how many blocks
>> shift - the bias imposed on the input
>> power - power of the item in the array
>> isExp - specify if we perform exp() on the input
*/
template <unsigned int goodSize> __global__
void KernelReduceSumFast(__half * input, __half * output, 
                         int stride, int strideNum, int reducedStrideNum, 
                         int blockSize, int blockNum,
                         __half * shift, __half power, bool isExp)
{
    unsigned int tid = threadIdx.y;
    unsigned int j = blockIdx.y * (blockDim.y * 2) + threadIdx.y;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= stride * blockNum)
        return;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    __shared__ __half iData[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ __half bias[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    if(threadIdx.y == 0)
        bias[threadIdx.x] = shift != NULL ? shift[i] : __float2half(0);
#else
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE bias[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    if(threadIdx.y == 0)
        bias[threadIdx.x] = shift != NULL ? __half2float(shift[i]) : 0;
#endif

    __syncthreads();

    /* first level reduction */
    int k = i / stride;
    int iOffset = i % stride;
    bool isValid = j < strideNum;
    bool isValid2 = j + blockDim.y < strideNum;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    __half * data =  iData + threadIdx.x * blockDim.y;
    __half * inputData = input  + k * blockSize;
    __half value  = isValid ? __hsub(inputData[j * stride + iOffset], bias[threadIdx.x]) : __float2half(0);
    __half value2 = isValid2 ? __hsub(inputData[(j + blockDim.y) * stride + iOffset], bias[threadIdx.x]) : __float2half(0);

    DTYPE powerf = __half2float(power);

    if(powerf != (DTYPE)1.0){
        if(powerf == (DTYPE)2.0){
            value = __hmul(value, value);
            value2 = __hmul(value2, value2);
        }
        else if(powerf == (DTYPE)0.5){
            value = hsqrt(value);
            value2 = hsqrt(value2);
        }
    }

    if(isExp){
        if(isValid)
            value = hexp(value);
        if(isValid2)
            value2 = hexp(value2);
    }

#else
    DTYPE * data =  iData + threadIdx.x * blockDim.y;
    __half * inputData = input  + k * blockSize;
    DTYPE value  = isValid ? __half2float(inputData[j * stride + iOffset]) - __half2float(bias[threadIdx.x]): 0;
    DTYPE value2 = isValid2 ? __half2float(inputData[(j + blockDim.y) * stride + iOffset]) - __half2float(bias[threadIdx.x]): 0;

    DTYPE powerf = __half2float(power);

    if(powerf != (DTYPE)1.0){
        if(powerf == (DTYPE)2.0){
            value = value * value;
            value2 = value2 *value2;
        }
        else if(powerf == (DTYPE)0.5){
            value = sqrt(value);
            value2 = sqrt(value2);
        }
        else{
            value = pow(value, powerf);
            value2 = pow(value2, powerf);
        }
    }

    if(isExp){
        if(isValid)
            value = exp(value);
        if(isValid2)
            value2 = exp(value2);
    }
#endif

    /* load data into the shared mem */
    data[tid] = value + value2;

    __syncthreads();

    /* unroll the warp */
    if(goodSize >= 512) {if(tid < 256) {data[tid] += data[tid + 256];} __syncthreads();}
    if(goodSize >= 256) {if(tid < 128) {data[tid] += data[tid + 128];} __syncthreads();}
    if(goodSize >= 128) {if(tid <  64) {data[tid] += data[tid +  64];} __syncthreads();}
    if(goodSize >= 64)  {if(tid <  32) {data[tid] += data[tid +  32];} __syncthreads();}
    if(goodSize >= 32)  {if(tid <  16) {data[tid] += data[tid +  16];} __syncthreads();}
    if(goodSize >= 16)  {if(tid <   8) {data[tid] += data[tid +   8];} __syncthreads();}
    if(goodSize >=  8)  {if(tid <   4) {data[tid] += data[tid +   4];} __syncthreads();}
    if(goodSize >=  4)  {if(tid <   2) {data[tid] += data[tid +   2];} __syncthreads();}
    if(goodSize >=  2)  {if(tid <   1) {data[tid] += data[tid +   1];} __syncthreads();}

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    /* write result for this block to the output array */
    if(threadIdx.y == 0 && blockIdx.y < reducedStrideNum) 
        output[(k * reducedStrideNum + blockIdx.y) * stride  + iOffset] = data[0];
#else
    /* write result for this block to the output array */
    if(threadIdx.y == 0 && blockIdx.y < reducedStrideNum) 
        output[(k * reducedStrideNum + blockIdx.y) * stride  + iOffset] = __float2half(data[0]);
#endif
}

/* 
sum the items along a dimension of the tensor (cuda version). 
For a 1-dimensional data array a,
sum = \sum_i (a_i - shift)^power if isExp == false
sum = \sum_i exp((a_i - shift)^power) if isExp == true
>> input - the input tensor
>> output - the output tensor
>> dim - which dimension to reduce
>> shift - the bias on the input
>> power - we perform pow(item_i, power) on each item
>> ieExp - specify if the exp() is performed
*/
void CudaReduceSumXT(XTensor * input, XTensor * output, int dim, XTensor * shift, DTYPE power, bool isExp)
{
    CheckNiuTransErrors((input && output), "Empty input or output tensors!");
    CheckNiuTransErrors((input->order == output->order + 1), "Incorrect tensor sizes!");
    CheckNiuTransErrors((input->order > dim && dim >=0), "Illegal dimension to reduce!");
    CheckNiuTransErrors((input->dataType == output->dataType), "Unmatched data types!");
    CheckNiuTransErrors((shift == NULL || output->unitNum == shift->unitNum), "Incorrect shift tensor size!");
    for(int i = 0; i < input->order; i++){
        if(i < dim){
            CheckNiuTransErrors((input->dimSize[i] == output->dimSize[i]), 
                                 "Unmatched tensors!");
        }
        else if(i > dim){
            CheckNiuTransErrors((input->dimSize[i] == output->dimSize[i - 1]), 
                                "Unmatched tensors!");
        }
    }

    if(input->dataType == X_FLOAT16){
        CheckNiuTransErrors((power == 0 || power == 0.5 || power == 1.0 || power == 2.0), "TODO!");
    }

    int cudaGridSize[3];
    int cudaBlockSize[3];
    int iter = 0;
    int stride = 1;
    int strideNum = input->dimSize[dim];
    int blockSize = 1;
    int blockNum = 1;

    for (int i = 0; i < input->order; i++) {
        if (i < dim)
            stride *= input->dimSize[i];
        else if (i > dim)
            blockNum *= input->dimSize[i];
    }
    blockSize = stride * strideNum;

    int devID = input->devID;
    XMem * mem = input->mem;

    GDevs->GetGridAndBlockSize2D(devID, strideNum, stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);

    int bufSize = input->unitSize * cudaGridSize[0] * stride * blockNum * 2;
    DTYPE * buf  = (DTYPE*)mem->AllocBuf(mem->devID, bufSize);
    DTYPE * buf1 = buf;
    DTYPE * buf2 = buf + cudaGridSize[0] * stride * blockNum;
    DTYPE * sp = shift != NULL ? (DTYPE*)shift->data : NULL;

    do{
        if(input->dataType == DTYPE_IN_MATRIX){
            DTYPE * iData = NULL;
            DTYPE * oData = NULL;
            if (iter == 0) {
                iData = (DTYPE*)input->data;
                oData = buf1;
            }
            else if (iter % 2 == 1) {
                iData = buf1;
                oData = buf2;
            }
            else {
                iData = buf2;
                oData = buf1;
            }
            /* unroll the reduction procedure. The code is messy but it is faster. */
            if(strideNum < 32){
                GDevs->GetGridAndBlockSize2D(devID, strideNum, stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (DTYPE*)output->data;
                KernelReduceSum <<<blocks, threads >>>(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum, sp, power, isExp);
            }
            else if(strideNum < 128){
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum/2+1, 64), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (DTYPE*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 64), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceSumFast<64> <<<blocks, threads >>>(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum, sp, power, isExp);
            }
            else if(strideNum < 256){
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum/2+1, 128), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (DTYPE*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 128), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceSumFast<128> <<<blocks, threads >>>(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum, sp, power, isExp);
            }
            else if(strideNum < 512){
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum/2+1, 256), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (DTYPE*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 256), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceSumFast<256> <<<blocks, threads >>>(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum, sp, power, isExp);
            }
            else{
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum/2+1, 512), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (DTYPE*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 512), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceSumFast<512> <<<blocks, threads >>>(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum, sp, power, isExp);
            }
        }
        else if(input->dataType == X_FLOAT16){
            __half * buf1ft16 = (__half *)buf1;
            __half * buf2ft16 = (__half *)buf2;
            __half * spft16 = (__half *)sp;
            unsigned short power2 = FloatToFloat16(power);
            __half * powerft16p = (__half*)&power2;
            __half * iData = NULL;
            __half * oData = NULL;
            if (iter == 0) {
                iData = (__half*)input->data;
                oData = buf1ft16;
            }
            else if (iter % 2 == 1) {
                iData = buf1ft16;
                oData = buf2ft16;
            }
            else {
                iData = buf2ft16;
                oData = buf1ft16;
            }

            /* unroll the reduction procedure. The code is messy but it is faster. */
            if(strideNum < 32){
                GDevs->GetGridAndBlockSize2D(devID, strideNum, stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (__half*)output->data;
                KernelReduceSum << <blocks, threads >> > (iData, oData, stride, strideNum, blocks.y, blockSize, blockNum, spft16, *powerft16p, isExp);
            }
            else if(strideNum < 128){
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum/2+1, 64), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (__half*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 64), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceSumFast<64> <<<blocks, threads >>>(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum, spft16, *powerft16p, isExp);
            }
            else if(strideNum < 256){
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum/2+1, 128), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (__half*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 128), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceSumFast<128> <<<blocks, threads >>>(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum, spft16, *powerft16p, isExp);
            }
            else if(strideNum < 512){
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum/2+1, 256), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (__half*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 256), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceSumFast<256> <<<blocks, threads >>>(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum, spft16, *powerft16p, isExp);
            }
            else{
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum/2+1, 512), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (__half*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 512), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceSumFast<512> <<<blocks, threads >>>(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum, spft16, *powerft16p, isExp);
            }
        }

        strideNum = cudaGridSize[0];
        blockSize = cudaGridSize[0];
        sp = NULL;
        power = (DTYPE)1.0;
        isExp = false;

        iter++;

    }while(strideNum > 1);

    mem->ReleaseBuf(mem->devID, bufSize);
}

/* 
reduce a tensor to another that keeps the max value along a dimension  - slow version

Given a block of data, we go over each dimension i in the stride and we have

sum_i = max_{0<=j<strideNum} input_{i,j}

where we can view the block as a matrix and input_{i,j} represent the item at the
crossing of the i-th columne and the j-th row.

>> input - the input array (representing a tensor)
>> output - the sum over each block. NOTE: output is also an array
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to finish the reduce
>> reducedStrideNum - the number of strides after reducation 
>> blockSize - size of the block (i.e., stride * strideNum)
>> blockNum - how many blocks
*/
 __global__
void KernelReduceMax(DTYPE * input, DTYPE * output, 
                     int stride, int strideNum, int reducedStrideNum, 
                     int blockSize, int blockNum)
{
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK * MIN_CUDA_SHARED_MEM_COL_SIZE/2];

    int idx = threadIdx.x * blockDim.y + threadIdx.y;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    if(i >= stride * blockNum)
        return;

    __syncthreads();

    int k = i / stride;
    int iOffset = i % stride;

    DTYPE value = (i < stride * blockNum && j < strideNum) ? 
                  input[blockSize * k + stride * j + iOffset]: FLOAT_MIN;

    /* load data into the shared mem */
    iData[threadIdx.x * blockDim.y + threadIdx.y] = value;

    __syncthreads();

    /* do reduction in shared mem */
    for (unsigned int s = blockDim.y/2; s > 0; s >>= 1){
        if(threadIdx.y < s && iData[idx] < iData[idx + s]){
            iData[idx] = iData[idx + s];
        }

        __syncthreads();
    }

    /* write result for this block to the output array */
    if (threadIdx.y == 0 && blockIdx.y < reducedStrideNum) 
        output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = iData[threadIdx.x * blockDim.y];

}

 /*
 reduce a tensor to another that keeps the max value along a dimension  - slow version

 Given a block of data, we go over each dimension i in the stride and we have

 sum_i = max_{0<=j<strideNum} input_{i,j}

 where we can view the block as a matrix and input_{i,j} represent the item at the
 crossing of the i-th columne and the j-th row.

 >> input - the input array (representing a tensor)
 >> output - the sum over each block. NOTE: output is also an array
 >> stride - stride that we need to move to the next item
 >> strideNum - how many strides we need to finish the reduce
 >> reducedStrideNum - the number of strides after reducation
 >> blockSize - size of the block (i.e., stride * strideNum)
 >> blockNum - how many blocks
 */
 __global__
 void KernelReduceMax(__half * input, __half * output,
         int stride, int strideNum, int reducedStrideNum,
         int blockSize, int blockNum)
 {
     int idx = threadIdx.x * blockDim.y + threadIdx.y;
     unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
     unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

     if (i >= stride * blockNum)
         return;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
     __shared__ __half iData[MAX_CUDA_THREAD_NUM_PER_BLOCK * MIN_CUDA_SHARED_MEM_COL_SIZE / 2];
#else
     __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK * MIN_CUDA_SHARED_MEM_COL_SIZE / 2];
#endif

     __syncthreads();

     int k = i / stride;
     int iOffset = i % stride;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
     __half value = (i < stride * blockNum && j < strideNum) ?
         input[blockSize * k + stride * j + iOffset] : __half(FLOAT16_MIN);
#else
     DTYPE value = (i < stride * blockNum && j < strideNum) ?
         __half2float(input[blockSize * k + stride * j + iOffset]) : FLOAT_MIN;
#endif

     /* load data into the shared mem */
     iData[threadIdx.x * blockDim.y + threadIdx.y] = value;

     __syncthreads();

     /* do reduction in shared mem */
     for (unsigned int s = blockDim.y / 2; s > 0; s >>= 1) {
         if (threadIdx.y < s && iData[idx] < iData[idx + s]) {
             iData[idx] = iData[idx + s];
         }

         __syncthreads();
     }

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
     /* write result for this block to the output array */
     if (threadIdx.y == 0 && blockIdx.y < reducedStrideNum)
         output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = iData[threadIdx.x * blockDim.y];
#else
     /* write result for this block to the output array */
     if (threadIdx.y == 0 && blockIdx.y < reducedStrideNum)
         output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = __half(iData[threadIdx.x * blockDim.y]);
#endif

 }


/* 
reduce a tensor to another that keeps the max value along a dimension  - fast version
>> input - the input array (representing a tensor)
>> output - the sum over each block. NOTE: output is also an array
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to finish the reduce
>> reducedStrideNum - the number of strides after reducation 
>> blockSize - size of the block (i.e., stride * strideNum)
>> blockNum - how many blocks
*/
template <unsigned int goodSize> __global__
void KernelReduceMaxFast(DTYPE * input, DTYPE * output, 
                         int stride, int strideNum, int reducedStrideNum, 
                         int blockSize, int blockNum)
{
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    unsigned int tid = threadIdx.y;
    unsigned int j = blockIdx.y * (blockDim.y * 2) + threadIdx.y;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= stride * blockNum)
        return;

    __syncthreads();

    /* first level reduction */
    int k = i / stride;
    int iOffset = i % stride;

    DTYPE * data =  iData + threadIdx.x * blockDim.y;
    DTYPE * inputData = input  + k * blockSize;
    DTYPE value  = j < strideNum ? inputData[j * stride + iOffset]: FLOAT_MIN;
    DTYPE value2 = j + blockDim.y < strideNum ? inputData[(j + blockDim.y) * stride + iOffset]: FLOAT_MIN;

    /* load data into the shared mem */
    data[tid] = MAX(value, value2);

    __syncthreads();

    /* unroll the warp */
    if(goodSize >= 512) {if(tid < 256) {if(data[tid] < data[tid + 256]) data[tid] = data[tid + 256];} __syncthreads();}
    if(goodSize >= 256) {if(tid < 128) {if(data[tid] < data[tid + 128]) data[tid] = data[tid + 128];} __syncthreads();}
    if(goodSize >= 128) {if(tid <  64) {if(data[tid] < data[tid +  64]) data[tid] = data[tid +  64];} __syncthreads();}
    if(goodSize >=  64) {if(tid <  32) {if(data[tid] < data[tid +  32]) data[tid] = data[tid +  32];} __syncthreads();}
    if(goodSize >=  32) {if(tid <  16) {if(data[tid] < data[tid +  16]) data[tid] = data[tid +  16];} __syncthreads();}
    if(goodSize >=  16) {if(tid <   8) {if(data[tid] < data[tid +   8]) data[tid] = data[tid +   8];} __syncthreads();}
    if(goodSize >=   8) {if(tid <   4) {if(data[tid] < data[tid +   4]) data[tid] = data[tid +   4];} __syncthreads();}
    if(goodSize >=   4) {if(tid <   2) {if(data[tid] < data[tid +   2]) data[tid] = data[tid +   2];} __syncthreads();}
    if(goodSize >=   2) {if(tid <   1) {if(data[tid] < data[tid +   1]) data[tid] = data[tid +   1];} __syncthreads();}

    /* write result for this block to the output array */
    if(threadIdx.y == 0 && blockIdx.y < reducedStrideNum) 
        output[(k * reducedStrideNum + blockIdx.y) * stride  + iOffset] = data[0];
}

/*
reduce a tensor to another that keeps the max value along a dimension  - fast version
>> input - the input array (representing a tensor)
>> output - the sum over each block. NOTE: output is also an array
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to finish the reduce
>> reducedStrideNum - the number of strides after reducation
>> blockSize - size of the block (i.e., stride * strideNum)
>> blockNum - how many blocks
*/
template <unsigned int goodSize> __global__
void KernelReduceMaxFast(__half * input, __half * output,
    int stride, int strideNum, int reducedStrideNum,
    int blockSize, int blockNum)
{
    unsigned int tid = threadIdx.y;
    unsigned int j = blockIdx.y * (blockDim.y * 2) + threadIdx.y;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= stride * blockNum)
        return;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    __shared__ __half iData[MAX_CUDA_THREAD_NUM_PER_BLOCK];
#else
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK];
#endif

    __syncthreads();

    /* first level reduction */
    int k = i / stride;
    int iOffset = i % stride;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    __half * data = iData + threadIdx.x * blockDim.y;
    __half * inputData = input + k * blockSize;
    __half value = j < strideNum ? inputData[j * stride + iOffset] : __half(FLOAT16_MIN);
    __half value2 = j + blockDim.y < strideNum ? inputData[(j + blockDim.y) * stride + iOffset] : __half(FLOAT16_MIN);
#else
    DTYPE * data = iData + threadIdx.x * blockDim.y;
    __half * inputData = input + k * blockSize;
    DTYPE value = j < strideNum ? __half2float(inputData[j * stride + iOffset]) : FLOAT_MIN;
    DTYPE value2 = j + blockDim.y < strideNum ? __half2float(inputData[(j + blockDim.y) * stride + iOffset]) : FLOAT_MIN;
#endif

    /* load data into the shared mem */
    data[tid] = MAX(value, value2);

    __syncthreads();

    /* unroll the warp */

    if (goodSize >= 512) { if (tid < 256) { if (data[tid] < data[tid + 256]) data[tid] = data[tid + 256]; } __syncthreads(); }
    if (goodSize >= 256) { if (tid < 128) { if (data[tid] < data[tid + 128]) data[tid] = data[tid + 128]; } __syncthreads(); }
    if (goodSize >= 128) { if (tid <  64) { if (data[tid] < data[tid +  64]) data[tid] = data[tid +  64]; } __syncthreads(); }
    if (goodSize >=  64) { if (tid <  32) { if (data[tid] < data[tid +  32]) data[tid] = data[tid +  32]; } __syncthreads(); }
    if (goodSize >=  32) { if (tid <  16) { if (data[tid] < data[tid +  16]) data[tid] = data[tid +  16]; } __syncthreads(); }
    if (goodSize >=  16) { if (tid <   8) { if (data[tid] < data[tid +   8]) data[tid] = data[tid +   8]; } __syncthreads(); }
    if (goodSize >=  8) { if (tid <    4) { if (data[tid] < data[tid +   4]) data[tid] = data[tid +   4]; } __syncthreads(); }
    if (goodSize >=  4) { if (tid <    2) { if (data[tid] < data[tid +   2]) data[tid] = data[tid +   2]; } __syncthreads(); }
    if (goodSize >=  2) { if (tid <    1) { if (data[tid] < data[tid +   1]) data[tid] = data[tid +   1]; } __syncthreads(); }

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    /* write result for this block to the output array */
    if (threadIdx.y == 0 && blockIdx.y < reducedStrideNum)
        output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = data[0];
#else
    /* write result for this block to the output array */
    if (threadIdx.y == 0 && blockIdx.y < reducedStrideNum)
        output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = __float2half(data[0]);
#endif
}

/*
reduce a tensor to another that keeps the max value along a dimension  - simple and fast version
*/
__global__
void KernelReduceMaxSimpleFast(DTYPE * input, DTYPE * output, 
                              int stride, int strideNum, int blockSize, int blockNum)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= stride)
        return;

    int blockIndex = i / blockSize;
    int offset = i % blockSize;

    DTYPE * ip = input + blockIndex * blockSize + offset;
    DTYPE * op = output + blockIndex * stride + offset;

    DTYPE max = DTYPE_MIN;
    if(strideNum % 4 == 0){
        int stride2 = stride + stride;
        int stride3 = stride2 + stride;
        int stride4 = stride3 + stride;
        for(int k = 0; k < blockSize; k += stride4){
            DTYPE m = MAX(MAX(ip[k], ip[k + stride]), MAX(ip[k + stride2], ip[k + stride3]));
            if(max < m)
                max = m;
        }
    }
    else{
        for(int k = 0; k < blockSize; k += stride)
            if(max < ip[k])
                max = ip[k];
    }

    __syncthreads();

    op[offset] = max;
}

/* 
get the max-valued items along a dimension of the tensor (cuda version). 
For a 1-dimensional data array a,

sum_i = max_{0<=j<strideNum} input_{i,j}

>> input - the input tensor
>> output - the output tensor
>> dim - which dimension to reduce
*/
void CudaReduceMaxXT(XTensor * input, XTensor * output, int dim)
{
    CheckNiuTransErrors((input && output), "Empty input or output tensors!");
    CheckNiuTransErrors((input->order == output->order + 1), "Incorrect tensor sizes!");
    CheckNiuTransErrors((input->order > dim && dim >=0), "Illegal dimension to reduce!");
    CheckNiuTransErrors((input->dataType == output->dataType), "Unmatched data types!");
    for(int i = 0; i < input->order; i++){
        if(i < dim){
            CheckNiuTransErrors((input->dimSize[i] == output->dimSize[i]), 
                                 "Unmatched tensors!");
        }
        else if(i > dim){
            CheckNiuTransErrors((input->dimSize[i] == output->dimSize[i - 1]), 
                                "Unmatched tensors!");
        }
    }

    int cudaGridSize[3];
    int cudaBlockSize[3];
    int iter = 0;
    int stride = 1;
    int strideNum = input->dimSize[dim];
    int blockSize = 1;
    int blockNum = 1;

    for (int i = 0; i < input->order; i++) {
        if (i < dim)
            stride *= input->dimSize[i];
        else if (i > dim)
            blockNum *= input->dimSize[i];
    }
    blockSize = stride * strideNum;

    int devID = input->devID;
    XMem * mem = input->mem;

    /*if(stride >= MAX_CUDA_THREAD_NUM_PER_BLOCK && strideNum > 200){
        GDevs->GetGridAndBlockSize(devID, stride, cudaGridSize, cudaBlockSize);
        KernelReduceMaxSimpleFast<<<cudaGridSize[0], cudaBlockSize[0]>>>
                                  ((DTYPE*)input->data, (DTYPE*)output->data, stride, strideNum,
                                    blockSize, blockNum);
        return;
    }*/

    GDevs->GetGridAndBlockSize2D(devID, strideNum, stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);

    int bufSize = sizeof(DTYPE) * cudaGridSize[0] * stride * blockNum * 2;
    DTYPE * buf  = (DTYPE*)mem->AllocBuf(mem->devID, bufSize);
    DTYPE * buf1 = buf;
    DTYPE * buf2 = buf + cudaGridSize[0] * stride * blockNum;

    do{
        if (input->dataType == DTYPE_IN_MATRIX) {
            DTYPE * iData = NULL;
            DTYPE * oData = NULL;
            if (iter == 0) {
                iData = (DTYPE*)input->data;
                oData = buf1;
            }
            else if (iter % 2 == 1) {
                iData = buf1;
                oData = buf2;
            }
            else {
                iData = buf2;
                oData = buf1;
            }

            /* unroll the reduction procedure. The code is messy but it is faster. */
            if (strideNum < 32) {
                GDevs->GetGridAndBlockSize2D(devID, strideNum, stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (DTYPE*)output->data;
                KernelReduceMax << <blocks, threads >> > (iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);
            }
            else if (strideNum < 128) {
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum / 2 + 1, 64), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (DTYPE*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 64), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceMaxFast<64> << <blocks, threads >> > (iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);
            }
            else if (strideNum < 256) {
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum / 2 + 1, 128), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (DTYPE*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 128), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceMaxFast<128> << <blocks, threads >> >(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);
            }
            else if (strideNum < 512) {
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum / 2 + 1, 256), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (DTYPE*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 256), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceMaxFast<256> << <blocks, threads >> >(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);
            }
            else {
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum / 2 + 1, 512), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (DTYPE*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 512), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceMaxFast<512> << <blocks, threads >> >(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);
            }
        }
        else if (input->dataType == X_FLOAT16) {
            __half * buf1ft16 = (__half *)buf1;
            __half * buf2ft16 = (__half *)buf2;
            __half * iData = NULL;
            __half * oData = NULL;
            if (iter == 0) {
                iData = (__half*)input->data;
                oData = buf1ft16;
            }
            else if (iter % 2 == 1) {
                iData = buf1ft16;
                oData = buf2ft16;
            }
            else {
                iData = buf2ft16;
                oData = buf1ft16;
            }

            /* unroll the reduction procedure. The code is messy but it is faster. */
            if (strideNum < 32) {
                GDevs->GetGridAndBlockSize2D(devID, strideNum, stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (__half*)output->data;
                KernelReduceMax << <blocks, threads >> >(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);
            }
            else if (strideNum < 128) {
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum / 2 + 1, 64), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (__half*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 64), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceMaxFast<64> << <blocks, threads >> >(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);
            }
            else if (strideNum < 256) {
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum / 2 + 1, 128), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (__half*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 128), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceMaxFast<128> << <blocks, threads >> > (iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);
            }
            else if (strideNum < 512) {
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum / 2 + 1, 256), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (__half*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 256), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceMaxFast<256> << <blocks, threads >> >(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);
            }
            else {
                GDevs->GetGridAndBlockSize2D(devID, MAX(strideNum / 2 + 1, 512), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                if (cudaGridSize[0] == 1)
                    oData = (__half*)output->data;
                CheckNiuTransErrors((cudaBlockSize[0] >= 512), "Incorrect thread number when calling the cuda kernel!");
                KernelReduceMaxFast<512> << <blocks, threads >> >(iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);
            }
        }
        
        strideNum = cudaGridSize[0];
        blockSize = cudaGridSize[0];

        iter++;

    }while(strideNum > 1);

    mem->ReleaseBuf(mem->devID, bufSize);
}

/* 
normalized the data with normal distribution (kernel code). For an input x,
y = a * (x-mean)/sqrt(variance+\epsilon) + b
where a and b are the scalar and bias respectively, and \epsilon is the adjustment parameter
>> input - the input data array
>> output - the output data array
>> mean - the mean of the input
>> var - the variance of the input
>> a - the scalar
>> b - the bias
>> epsilon - a parameter
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to go over for next block
>> blockNum - how many blocks we have
*/
__global__
void KernelNormalize(DTYPE * input, DTYPE * output, DTYPE * mean, DTYPE * var,
                     DTYPE * a, DTYPE * b, DTYPE epsilon, 
                     int stride, int strideNum, int blockNum)
{
    __shared__ DTYPE iMean[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE iVar[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int iBlock[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int iOffset[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int blockSize;

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= stride * blockNum || j >= strideNum)
        return;

    if(threadIdx.y == 0){
        iOffset[threadIdx.x] = i % stride;
        iBlock[threadIdx.x] = i / stride;
        iMean[threadIdx.x] = mean[i];
        iVar[threadIdx.x] = var[i];
        blockSize = stride * strideNum;
    }

    __syncthreads();

    int inBlockOffset = j * stride + iOffset[threadIdx.x];
    int offset = iBlock[threadIdx.x] * blockSize + inBlockOffset;

    output[offset] = a[inBlockOffset] * (input[offset] - iMean[threadIdx.x])/sqrt(iVar[threadIdx.x] + epsilon) + b[inBlockOffset];
}

/* 
normalized the data with normal distribution. For an input x,
y = a * (x-mean)/sqrt(variance+\epsilon) + b
where a and b are the scalar and bias respectively, and \epsilon is the adjustment parameter
>> input - the input tensor
>> output - the output tensor
>> dim - dimension alone which we generate the mean and variance
>> mean - the mean of the input
>> var - the variance of the input
>> a - the scalar
>> b - the bias
>> epsilon - a parameter
*/
extern "C" 
void CudaNormalize(XTensor * input, XTensor * output, int dim, 
                   XTensor * mean, XTensor * var, 
                   XTensor * a, XTensor * b, 
                   DTYPE epsilon)
{
    CheckNiuTransErrors((input->dataType == DTYPE_IN_MATRIX), "TODO!");

    int stride = 1;
    int strideNum = input->dimSize[dim];
    int blockNum = 1;
    for (int i = 0; i < input->order; i++) {
        if(i < dim)
            stride *= input->dimSize[i];
        else if(i > dim)
            blockNum *= input->dimSize[i];
    }

    int cudaGridSize[3];
    int cudaBlockSize[3];

    GDevs->GetGridAndBlockSize2D(input->devID, strideNum, stride * blockNum, 
                                 MAX_INT, cudaGridSize, cudaBlockSize);

    dim3 blocks(cudaGridSize[1], cudaGridSize[0]);
    dim3 threads(cudaBlockSize[1], cudaBlockSize[0]);

    KernelNormalize<<<blocks, threads>>>((DTYPE*)input->data, (DTYPE*)output->data, 
                                         (DTYPE*)mean->data, (DTYPE*)var->data, 
                                         (DTYPE*)a->data, (DTYPE*)b->data, epsilon,
                                          stride, strideNum, blockNum);
}

#endif