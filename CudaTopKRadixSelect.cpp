void CudaTopKRadixSelect(XTensor * a, XTensor * b, XTensor * index, int dim, int k)
{
	CheckNiuTransErrors((a->unitSize == b->unitSize), "Unmatched input tensors!");
	CheckNiuTransErrors((a->order == b->order), "Unmatched input tensors!");
	CheckNiuTransErrors((index == NULL || a->order == index->order), "Unmatched input tensors!");
	CheckNiuTransErrors((index->dataType == X_INT), "Wrong data type!");
	CheckNiuTransErrors((b->dimSize[dim] == k), "A too large K");
	// Tensor(100, 200, 300, 400, 500, 600)  dim = 2
	// thread(100x200x400x500x600, 300)
	int stride = 1;
	int strideNumA = a->dimSize[dim];
	for (int i = 0; i < dim; i++)
		stride *= a->dimSize[i];

	int blockNum = 1;
	for (int i = dim + 1; i < a->order; i++)
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
		if (a->dataType == DTYPE_IN_MATRIX) {
			float* input = (float*)malloc(sizeof(float)*strideNumA*blockNum);
			cudaMemcpy(input, a->data, sizeof(float)*strideNumA*blockNum, cudaMemcpyDeviceToHost);
			unsigned int* output = (unsigned int*)malloc(sizeof(unsigned int)*strideNumA*blockNum*stride), *goutput;
			cudaMalloc(&goutput, sizeof(unsigned int)*strideNumA*blockNum*stride);
			convert2uint << <1, 1 >> >((float*)a->data, goutput, strideNumA*blockNum*stride);
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
