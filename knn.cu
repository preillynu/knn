
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

//Set grid and block size for the kernels that run and sets the number of neighbors desired
int k=7;
int gridSize = 1000;
int blockSize = 1024;

//The distKernel computes the difference squared between two points. Requires size number of threads
__global__ void distKernel(float *a, const float coor, const int size)
{
	//Get the Thread ID
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	//Only calculate the distance if the thread corresponds to an existing element
	if (i < size){
		a[i] = (a[i] - coor)*(a[i] - coor);
	}
}

int main()
{
	//Input file to read the labeled data
	ifstream inputData;
	
	//The arrays that will hold the labeled data
	float* xcoors;
	float* ycoors;
	
	//Arrays to hold the distances computed via the GPU and CPU
	float* distances;
	float* distances2;
	
	//An array to hold the unsorted labels
	int* unsortedlabels;
	
	//Variable used to label new data; defaults to 0
	int outputLabel = 0;
	
	//Variables used to hold points of interest
	float x;
	float y;
	
	//Variables to hold the number of labeled data points and the number or points being entered
	int numInput;
	int numPoints;
	
	//Opens a file whose first line is the number of elements and every subsequent line is an x coordinate, y coordinate,
	//and label that are seperated by spaces
	inputData.open("Synthesized_Data_1M.txt");

	//Make sure the file opens correctly
	if (!inputData.is_open()){
		cout << "Something went wrong while reading in the data. Check where it is located again." << endl;
		exit(0);
	}
	
	//Prompt the user for the number of points being classified
	cout << "How many points do you want to read? ";
	cin >> numPoints;

	//Store the number of labeled data points
	inputData >> numInput;
	
	//Set up the arrays to have a max capacity equal to the sum of the number of labeled and unlabeled points
	xcoors = new float[numInput + numPoints];
	ycoors = new float[numInput + numPoints];
	unsortedlabels = new int[numInput + numPoints];
	distances = new float[numInput + numPoints];
	distances2 = new float[numInput + numPoints];
	//Set up pointers for the arrays used for the GPU implementation 
	float *devX, *devY;
	
	for (int i = 0; i < numInput; i++){
		//Begin modifying data to find the distance more easily as no scalar - vector CUBLAS sum function
		inputData >> x >> y >> unsortedlabels[i];
		xcoors[i] = x;
		ycoors[i] = y;

	}
	
	//Close the input file
	inputData.close();

	//Collect the data points that the user wants classified
	for (int i = 0; i < numPoints; i++){
		cout << i << " data point: "<< endl;
		cout << "x-coordinate: ";
		cin >> xcoors[i+numInput];
		cout << "y-coordinate: ";
		cin >> ycoors[i + numInput];
		cout << endl;
	}

	//Run the KNN distance finding and sorting code for however many points the user entered
	for (int z = 0; z < numPoints; z++){
		//Get the coordinates of the point to be classified
		x = xcoors[numInput + z];
		y = ycoors[numInput + z];
		cout << z << " data point: " << endl;

		//Create an array to hold labels that will be sorted
		int* labels;
		labels = new int[numInput + z];

		//Copy all of the labels to the new array
		for (int i = 0; i < numInput + z; i++){
			labels[i] = unsortedlabels[i];
		}

		//Allocate and fill the arrays for the GPU version
		cudaMalloc((void**)&devX, (numInput + z)*sizeof(float));
		cudaMalloc((void**)&devY, (numInput + z)*sizeof(float));

		cudaMemcpy(devX, xcoors, (numInput + z)*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(devY, ycoors, (numInput + z)*sizeof(float), cudaMemcpyHostToDevice);

		//Time the sequential version using Windows's QueryPerfomanceCounter()
		
		//Number of ticks per second
		LARGE_INTEGER frequency;    
		//Measure times
		LARGE_INTEGER t1, t2;  
		//Store time
		double elapsedTime;

		//Fill the frequency variable
		QueryPerformanceFrequency(&frequency);

		//Get the first time
		QueryPerformanceCounter(&t1);

		for (int i = 0; i < numInput+z; i++){
			//Compute the distances using the CPU
			distances2[i] = (xcoors[i] - x)*(xcoors[i] - x) + (ycoors[i] - y)*(ycoors[i] - y);
		}
		//Get the second time
		QueryPerformanceCounter(&t2);

		//Get the elapsed time in milliseconds
		elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
		cout << elapsedTime << " milliseconds for sequential run." << endl;

		//Create cuBlas handles to use cuBlas
        	cublasHandle_t h;
        	cublasCreate(&h);

		//Set up and alpha for saxpy()
        	float alpha = 1.0f;

		//Create CUDA Events to time the GPU version
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		//Start timer
		cudaEventRecord(start);
		
		//Get the x and y squared differences
		distKernel << <blockSize, gridSize >> >(devX, x, numInput+z);
		distKernel << <blockSize, gridSize >> >(devY, y, numInput+z);
		
		//Sum the squared differences using cuBlas
		cublasSaxpy(h, numInput+z, &alpha, devX, 1, devY, 1);
		cudaEventRecord(stop);
		
		//Find the time for the GPU version and print it
        	cudaEventSynchronize(stop);
        	float milliseconds = 0;
        	cudaEventElapsedTime(&milliseconds, start, stop);
        	cout << milliseconds << " milliseconds for parallel run." << endl;
		
		//Copy the GPU computed distances over to the host
        	cudaMemcpy(distances, devY, (numInput+z)*sizeof(float), cudaMemcpyDeviceToHost);
	
		//Get rid of the cuBlas handle
		cublasDestroy(h);

		//Calculate the number of distances that were computed differently by the CPU and GPU
		int numWrong = 0;
		for (int i = 0; i < numInput + z; i++){
			if (distances2[i] != distances[i])
				numWrong++;
		}
		
		//Print a message if any distances were incorrectly computed
		if (numWrong > 0)
			cout << numWrong << " distances miscomputed\n";

		//Free the CUDA Arrays
		cudaFree(devX);
		cudaFree(devY);

		//Use the thrust library to sort the distances and theit corresponding labels
		thrust::sort_by_key(distances, distances + numInput + z, labels);

		//Count the number of points labeled 0 and 1, telling the user the label of all k-nearest neighbors
		int count1 = 0;
		for (int i = 0; i < k; i++){
			if (labels[i] == 0){
				count1++;
			}

			cout << "" << i + 1 << " closest point has a label of: " << labels[i] << " with a squared distance of " << distances[i] << endl;
		}

		//Choose the output label based on the labels of the k-nearest-neighbors
		if (k-count1 > count1)
			outputLabel = 1;
		else{
			outputLabel = 0;
		}

		//Output the classification for the data point
		cout << "(" << x << "," << y << ") should be classified as: " << outputLabel << endl;
		cout << endl;
		
		//Add the points label to the unsorted labels
		unsortedlabels[z + numInput] = outputLabel;
		//Free the sorted array of labels
		free(labels);
	}


    	// cudaDeviceReset must be called before exiting in order for profiling and
    	// tracing tools such as Nsight and Visual Profiler to show complete traces.
    	cudaDeviceReset();

	//Create an ofstream to print the data to so that it can be used as a labeled data set for another run
	ofstream outputData;
	
	//Open a file and make sure it is empty before writing to it
	outputData.open("Synthesized_Data_Updated.txt", ios::out | ios::trunc);
	
	//Make sure the file opens correctly
	if (!outputData.is_open()){
		cout << "Something went wrong with opening the output file. Check where it is located again." << endl;
		exit(0);
	}
	
	//Put the total number of data points at the top of the file
	outputData << (numInput + numPoints) << endl;
	
	//Print each point and its correspoding label
	for (int i = 0; i < numInput + numPoints; i++){
		outputData << xcoors[i] << " " << ycoors[i] << " " << unsortedlabels[i] << endl;
	}
	
	//Close the file once it is written
	outputData.close();

	//Free remaining arrays
	free(xcoors);
	free(ycoors);
	free(unsortedlabels);
	free(distances);
	free(distances2);

	//Pause on Windows machines to view output
	system("pause");
    return 0;
}
