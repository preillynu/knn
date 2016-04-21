
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <thrust/sort.h>
#include <stdio.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <Windows.h>
using namespace std;

int numInput;
float x;
float y;
int k=7;
int gridSize = 1000;
int blockSize = 1024;
int numPoints = 1;

__global__ void distKernel(float *a, const float coor, const int size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size){
		a[i] = (a[i] - coor)*(a[i] - coor);
	}
}

int main()
{
	ifstream inputData;
	float* xcoors;
	float* ycoors;
	float* distances;
	float* distances2;
	int* unsortedlabels;
	int outputLabel = 0;

	inputData.open("Synthesized_Data_1M.txt");

	if (!inputData.is_open()){
		cout << "Something went wrong while reading in the data. Check where it is located again." << endl;
		exit(0);
	}

	cout << "How many points do you want to read? ";
	cin >> numPoints;

	inputData >> numInput;
	xcoors = new float[numInput + numPoints];
	ycoors = new float[numInput + numPoints];
	unsortedlabels = new int[numInput + numPoints];
	distances = new float[numInput + numPoints];
	distances2 = new float[numInput + numPoints];
	float *devX, *devY;

	float xcoor = 0.0;
	float ycoor = 0.0;

	for (int i = 0; i < numInput; i++){
		//Begin modifying data to find the distance more easily as no scalar - vector CUBLAS sum function
		inputData >> xcoor >> ycoor >> unsortedlabels[i];
		xcoors[i] = xcoor;
		ycoors[i] = ycoor;

	}
	
	inputData.close();

	for (int i = 0; i < numPoints; i++){
		cout << i << " data point: "<< endl;
		cout << "x-coordinate: ";
		cin >> xcoors[i+numInput];
		cout << "y-coordinate: ";
		cin >> ycoors[i + numInput];
		cout << endl;
	}
 
	cudaDeviceSynchronize();

	for (int z = 0; z < numPoints; z++){
		x = xcoors[numInput + z];
		y = ycoors[numInput + z];
		cout << z << " data point: " << endl;

		int* labels;
		labels = new int[numInput + z];

		for (int i = 0; i < numInput + z; i++){
			labels[i] = unsortedlabels[i];
		}

		cudaMalloc((void**)&devX, (numInput + z)*sizeof(float));
		cudaMalloc((void**)&devY, (numInput + z)*sizeof(float));

		cudaMemcpy(devX, xcoors, (numInput + z)*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(devY, ycoors, (numInput + z)*sizeof(float), cudaMemcpyHostToDevice);

		LARGE_INTEGER frequency;        // ticks per second
		LARGE_INTEGER t1, t2;           // ticks
		double elapsedTime;

		// get ticks per second
		QueryPerformanceFrequency(&frequency);

		// start timer
		QueryPerformanceCounter(&t1);

		for (int i = 0; i < numInput+z; i++){
			//Compute the distances manually
			distances2[i] = (xcoors[i] - x)*(xcoors[i] - x) + (ycoors[i] - y)*(ycoors[i] - y);
		}
		QueryPerformanceCounter(&t2);

		// compute and print the elapsed time in millisec
		elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
		cout << elapsedTime << " milliseconds for sequential run." << endl;

        cublasHandle_t h;
        cublasCreate(&h);

        float alpha = 1.0f;

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		distKernel << <blockSize, gridSize >> >(devX, x, numInput+z);
		distKernel << <blockSize, gridSize >> >(devY, y, numInput+z);
		cudaEventRecord(stop);

		cublasSaxpy(h, numInput+z, &alpha, devX, 1, devY, 1);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        cout << milliseconds << " milliseconds for parallel run." << endl;

        cudaDeviceSynchronize();
        cudaMemcpy(distances, devY, (numInput+z)*sizeof(float), cudaMemcpyDeviceToHost);

		cublasDestroy(h);

		int numWrong = 0;
		for (int i = 0; i < numInput + z; i++){
			//Compute the distances manually
			if (distances2[i] != distances[i])
				numWrong++;
		}
		if (numWrong > 0)
			cout << numWrong << " distances miscomputed\n";

		cudaFree(devX);
		cudaFree(devY);

		thrust::sort_by_key(distances, distances + numInput + z, labels);

		int count1 = 0;
		int count2 = 0;
		for (int i = 0; i < k; i++){
			if (labels[i] == 0){
				count1++;
			}
			else{
				count2++;
			}
			cout << "" << i + 1 << " closest point has a label of: " << labels[i] << " with a distance of " << distances[i] << endl;
		}

		if (count2 > count1)
			outputLabel = 1;
		else{
			outputLabel = 0;
		}

		cout << "(" << x << "," << y << ") should be classified as: " << outputLabel << endl;
		cout << endl;
		unsortedlabels[z + numInput] = outputLabel;
		free(labels);
	}


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();

	ofstream outputData;
	outputData.open("Synthesized_Data_Updated.txt", ios::out | ios::trunc);
	if (!outputData.is_open()){
		cout << "Something went wrong with opening the output file. Check where it is located again." << endl;
		exit(0);
	}
	outputData << (numInput + numPoints) << endl;
	for (int i = 0; i < numInput + numPoints; i++){
		outputData << xcoors[i] << " " << ycoors[i] << " " << unsortedlabels[i] << endl;
	}
	outputData.close();

	free(xcoors);
	free(ycoors);
	free(unsortedlabels);
	free(distances);
	free(distances2);

	system("pause");
    return 0;
}
