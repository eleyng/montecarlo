// Written by Barry Wilkinson, UNC-Charlotte. Pi.cu  December 22, 2010.
//Derived somewhat from code developed by Patrick Rogers, UNC-C

// edits and additional comments for clarity by George Gorospe.
// george.e.gorospe@nasa.gov

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

#define TRIALS_PER_THREAD 4096
#define BLOCKS 256
#define THREADS 256
#define PI 3.1415926535  // known value of pi


// Device Code, this code operates on the GPU
// inputs: device memory pointer "estimate"
//	   state of the CURAND PRNG device memory pointer
__global__ void gpu_monte_carlo(float *estimate, curandState *states) {

	__shared__ float cache[THREADS]; 
	
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	
	// Holds the count of points in the circle
	// each thread performs 4096 trials 
	int points_in_circle = 0;
	float x, y;
	
	// Call to create CURAND list of random numbers
	// first input: unsigned long long seed: used for init state
	// second input: unsigned long long sequence: 
	// thrid input: unsigned long long offset
	// fourth input curandState_t *state
	curand_init(1234, tid, 0, &states[tid]);  // 	Initialize CURAND

	// Creating random values for x and y then comparing to the circle
	// curand_uniform: this function retruns a sequence of pseudorandom 
	// 	floats uniformly distributed between 0.0 and 1.0.
	
	// Create temporary array to store estimates generated from threads

	for(int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = curand_uniform (&states[tid]);
		y = curand_uniform (&states[tid]);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
		
	}
	estimate[tid] = 4.0f * points_in_circle / (float) TRIALS_PER_THREAD; // return estimate of pi
	
	if (tid < (THREADS * BLOCKS)) { cache[threadIdx.x] = estimate[tid]; }
	
	__syncthreads();
	
	for (int stride = blockDim.x/2; stride > 0 ; stride /= 2) {
		if (threadIdx.x < stride)
			cache[threadIdx.x] += cache[threadIdx.x + stride];
		__syncthreads();
	}
	
	if (threadIdx.x == 0) {
		estimate[blockIdx.x] = cache[0] / THREADS;
	}
}

float host_monte_carlo(long trials) {
	float x, y;
	long points_in_circle;
	for(long i = 0; i < trials; i++) {
		x = rand() / (float) RAND_MAX;
		y = rand() / (float) RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0f);
	}
	return 4.0f * points_in_circle / trials;
}

int main (int argc, char *argv[]) {
	clock_t start, stop;
	float host[BLOCKS * THREADS];
	float *dev;
	curandState *devStates;

	/*printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", TRIALS_PER_THREAD,
BLOCKS, THREADS);
	printf("Total number of samples: %d.\n", TRIALS_PER_THREAD*THREADS*BLOCKS);
	*/
	start = clock();

	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(float)); // allocate device mem. for counts
	
	cudaMalloc( (void **)&devStates, THREADS * BLOCKS * sizeof(curandState) );

	gpu_monte_carlo<<<BLOCKS, THREADS>>>(dev, devStates);

	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost); // return results 

	float pi_gpu;
	
	//average the estimates
	for(int i = 0; i < BLOCKS; i++) {
		pi_gpu += host[i];
	}

	pi_gpu /= (BLOCKS);

	stop = clock();

	printf(/*"GPU pi calculated in %f s.\n",*/ "%f ",  (stop-start)/(float)CLOCKS_PER_SEC);

	start = clock();
	float pi_cpu = host_monte_carlo(BLOCKS * THREADS * TRIALS_PER_THREAD);
	stop = clock();
	printf(/*"CPU pi calculated in %f s.\n",*/ "%f ", (stop-start)/(float)CLOCKS_PER_SEC);

	printf(/*"CUDA estimate of PI = %f [error of %f]\n",*/ "%f ",/* pi_gpu,*/ pi_gpu - PI);
	printf(/*"CPU estimate of PI = %f [error of %f]\n",*/ "%f \n", /* pi_cpu,*/ pi_cpu - PI);
	
	return 0;
}

