

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <stdio.h>


#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <Windows.h>
using namespace std;


//Set grid and block size for the kernels that run and sets the number of neighbors desired
int k = 7;
int gridSize = 100;
int blockSize = 1024;

//The distKernel computes the difference squared between two points. Requires size number of threads
__global__ void distKernel(float *coor, float *dist, float* p, const int size, const int dims)
{
	//Get the Thread ID
	int id = (threadIdx.x+ blockIdx.x * blockDim.x)*dims;

	//Only calculate the distance if the thread corresponds to an existing element
	if (id < size*dims){
		dist[(id/dims)] = 0.0;
		for (int i = 0; i < dims; i++){
			dist[(id/dims)] += (coor[id+i] - p[i])*(coor[id+i] - p[i]);
		}
	}
}

int main()
{
	//Input file to read the labeled data
	ifstream inputData;

	//The arrays that will hold the labeled data
	float* coors;

	//Arrays to hold the distances computed via the GPU and CPU
	float* distances;
	float* distances2;

	//An array to hold the unsorted labels
	int* unsortedlabels;

	//Variable used to label new data; defaults to 0
	int *outputLabel;

	//Variables used to hold one data point
	float* dataPoint;

	//Variables to hold the number of labeled data points and the number or points being entered
	int numInput;
	int numPoints;
	int numDim;
	int numLabel;

	//Opens a file whose first line is the number of elements and every subsequent line is an x coordinate, y coordinate,
	//and label that are seperated by spaces
	inputData.open("Customized_Data_Updated.txt");

	//Make sure the file opens correctly
	if (!inputData.is_open()){
		cout << "Something went wrong while reading in the data. Check where it is located again." << endl;
		exit(0);
	}

	//Prompt the user for the number of points being classified
	cout << "How many points do you want to read? ";
	cin >> numPoints;

	//Store the number of labeled data points, the number of dimensions and the total number of labels
	inputData >> numInput >> numDim >> numLabel;

	//Set up the arrays to have a max capacity equal to the sum of the number of labeled and unlabeled points
	coors = new float[numDim*(numInput + numPoints)];
	unsortedlabels = new int[numInput + numPoints];
	distances = new float[numInput + numPoints];
	distances2 = new float[numInput + numPoints];
	dataPoint = new float[numDim];
	//Set up pointers for the arrays used for the GPU implementation 
	float *devX, *devD, *devP;

	//Set up the proper grid size
	gridSize = (numInput + numPoints) / blockSize + 1;

	for (int i = 0; i < numInput; i++){
		//Begin modifying data to find the distance more easily as no scalar - vector CUBLAS sum function
		for (int j = 0; j < numDim; j++){
			inputData >> coors[i*numDim + j];
		}
		inputData >> unsortedlabels[i];
	}

	//Close the input file
	inputData.close();

	//Collect the data points that the user wants classified
	for (int i = 0; i < numPoints; i++){
		cout << i << " data point: " << endl;
		for (int j = 0; j < numDim; j++){
			cout << j << " dim: ";
			cin >> coors[(i + numInput)*numDim + j];
		}
		cout << endl;
	}

	//Run the KNN distance finding and sorting code for however many points the user entered
	for (int z = 0; z < numPoints; z++){

		//Get the coordinates of the point to be classified
		for (int i = 0; i < numDim; i++){
			dataPoint[i] = coors[numDim*(numInput + z) + i];
		}
		cout << z << " data point: " << endl;

		//Create an array to hold labels that will be sorted
		int* labels;
		labels = new int[numInput + z];

		//Copy all of the labels to the new array
		for (int i = 0; i < numInput + z; i++){
			labels[i] = unsortedlabels[i];
		}

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

		for (int i = 0; i < numInput + z; i++){
			//Compute the distances using the CPU
			distances2[i] = 0.0;
			for (int j = 0; j < numDim; j++){
				distances2[i] += (coors[i*numDim + j] - dataPoint[j])*(coors[i*numDim + j] - dataPoint[j]);
			}
		}

		//Get the second time
		QueryPerformanceCounter(&t2);

		//Get the elapsed time in milliseconds
		elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
		cout << elapsedTime << " milliseconds for sequential run." << endl;

		//Allocate and fill the arrays for the GPU version
		cudaMalloc((void**)&devX, (numInput + z)*numDim*sizeof(float));
		cudaMalloc((void**)&devD, (numInput + z)*sizeof(float));
		cudaMalloc((void**)&devP, (numDim)*sizeof(float));

		//Create CUDA Events to time the GPU version
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		//Start timer
		cudaEventRecord(start);

		//Copy in the data
		cudaMemcpy(devX, coors, (numInput + z)*numDim*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(devD, coors, (numInput + z)*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(devP, dataPoint, (numDim)*sizeof(float), cudaMemcpyHostToDevice);


		//Compute the distances for the next dimension
		distKernel<<<blockSize, gridSize>>>(devX, devD, devP, numInput + z, numDim);

		//Finish timing
		cudaEventRecord(stop);

		//Find the time for the GPU version and print it
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		cout << milliseconds << " milliseconds for parallel run." << endl;

		//Copy the GPU computed distances over to the host
		cudaMemcpy(distances, devD, (numInput + z)*sizeof(float), cudaMemcpyDeviceToHost);

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
		cudaFree(devD);

		//Use the thrust library to sort the distances and theit corresponding labels
		thrust::sort_by_key(distances, distances + numInput + z, labels);

		//Set up an array to hold the number of k-nearest-neighbors with that label
		int* labelCounts = new int[numLabel];
		for (int i = 0; i < numLabel; i++){
			labelCounts[i] = 0;
		}

		//Count the number of points labeled 0 and 1, telling the user the label of all k-nearest neighbors
		for (int i = 0; i < k; i++){
			labelCounts[labels[i]] += 1;
			cout << "" << i + 1 << " closest point has a label of: " << labels[i] << " with a squared distance of " << distances[i] << endl;
		}

		//Find the correct output label
		outputLabel = thrust::max_element(thrust::host, labelCounts, labelCounts + k);
		int output = outputLabel - labelCounts;

		//Output the classification for the data point
		cout << "Point " << z << " should be classified as: " << output << endl;
		cout << endl;

		//Add the points label to the unsorted labels
		unsortedlabels[z + numInput] = output;
		//Free the sorted array of labels
		free(labels);
		free(labelCounts);
	}


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaDeviceReset();

	//Create an ofstream to print the data to so that it can be used as a labeled data set for another run
	ofstream outputData;

	//Open a file and make sure it is empty before writing to it
	outputData.open("Customizedb_Data_Updated.txt", ios::out | ios::trunc);

	//Make sure the file opens correctly
	if (!outputData.is_open()){
		cout << "Something went wrong with opening the output file. Check where it is located again." << endl;
		exit(0);
	}

	//Put the total number of data points at the top of the file
	outputData << (numInput + numPoints) << " " << numDim << " " << numLabel << endl;

	//Print each point and its correspoding label
	for (int i = 0; i < numInput + numPoints; i++){
		for (int j = 0; j < numDim; j++){
			outputData << coors[i*numDim + j] << " " << endl;
		}
		outputData << unsortedlabels[i] << endl;
	}

	//Close the file once it is written
	outputData.close();
	free(coors);
	free(dataPoint);
	free(unsortedlabels);
	free(distances);
	free(distances2);

	//Pause on Windows machines to view output
	system("pause");
	return 0;
}
