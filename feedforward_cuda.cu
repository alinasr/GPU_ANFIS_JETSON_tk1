/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
//#include "helper_cuda.h"
//#include "feedforward_kernel.h"
//#include "feedforward_kernel.cuh"

// The feedforward CUDA GPU thread function

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include<math.h>


// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>



#include "loaddata.h"
//#include "properties.h"


//int32_t* inputs;
//int32_t* output;

int32_t* rule_list_host;



#define mult_cuda(ans,a,b) ans=((int64_t)a*(int64_t)b)>>frac
#define divide_cuda(ans,a,b) ans=((int64_t)a<<frac)/b
#define div2_cuda(a,b) (((int64_t)a<<frac)/b)
#define pow2_cuda(ans,a) ans=((int64_t)a*(int64_t)a)>>frac 
#define fixsqr_cuda(a) (sqrt((int64_t)a<<frac))
#define p2_cuda(a) ((int64_t)(((int64_t)a*(int64_t)a)>>frac))
#define p4_cuda(a) p2_cuda(p2_cuda(a))
#define intpow_cuda(a,b) ((int64_t)pow(a,(b>>frac)))>>((b>>frac) - 1)*frac
//#define gussM(x,sig,c) one - ((div2((p2((x-c))),(p2(sig))))>>3)
#define gussM_cuda(x,sig,c) (div2_cuda((p2_cuda(sig)),((p2_cuda(sig)) + (p2_cuda((x-c))))))




#define frac 24
#define one pow((double) 2,(double)frac)
#define two one*2
#define fore one*4
#define three one*3
#define five one*5
#define six one*6

//__device__ int32_t*  rule_list_cuda;


int32_t* inputs1;
int32_t* output1;

__device__ __constant__ int32_t data_len;
__device__  __constant__ int32_t number_of_inputs;
__device__  __constant__ int32_t number_of_mffunc;
__device__  __constant__ int32_t mflayrs;
__device__ __constant__ int32_t nods;

 int32_t h_data_len;
 int32_t h_number_of_inputs;
 int32_t h_number_of_mffunc;
 int32_t h_mflayrs;
 int32_t h_nods;


int32_t test_len;
int32_t fixdata_len;
int32_t fixtest_len;
int32_t  rules_numb;
int32_t paramnum;
int32_t flayrs;
int32_t SCsize;
int32_t tempsize;

extern "C"
void intial_cu(const properties& p1, int32_t *x, int32_t *y) {

	

	 h_data_len = p1.data_len;
	 h_number_of_inputs = p1.number_of_inputs;;
	 h_number_of_mffunc = p1.number_of_mffunc;
	 h_mflayrs = h_number_of_mffunc * h_number_of_inputs;
	 h_nods= p1.number_of_rules;

	
	 cudaMemcpyToSymbol(data_len, &h_data_len, sizeof(int32_t), 0, cudaMemcpyHostToDevice);
	 cudaMemcpyToSymbol(number_of_inputs, &h_number_of_inputs, sizeof(int32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(number_of_mffunc, &h_number_of_mffunc,  sizeof(int32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(mflayrs, &h_mflayrs, sizeof(int32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(nods, &h_nods, sizeof(int32_t), 0, cudaMemcpyHostToDevice);
	


	 flayrs = (h_number_of_inputs + 1)*h_nods;

	 SCsize = h_number_of_inputs*h_number_of_mffunc * 2;     //guass 
	 tempsize = h_nods*(h_number_of_inputs + 1);

	 inputs1 = x;
	 output1 = y;

	
	 rule_list_host = new (std::nothrow) int32_t[nods*number_of_inputs];


	 if (h_nods == pow(h_number_of_mffunc, h_number_of_inputs)) {
		 for (int i = 0; i < h_nods; ++i) {
			 int temp;
			 temp = i;
			 for (int j = 0; j < h_number_of_inputs; ++j) {

				 rule_list_host[i*(h_number_of_inputs)+j] = (temp%h_number_of_mffunc) + h_number_of_mffunc*(h_number_of_inputs - 1 - j);

				 temp = temp / h_number_of_mffunc;

			 }

		 }

	 }



}



__device__ int32_t gaussmf(int32_t x, int32_t sig, int32_t c) {



float temp1;
int32_t temp2;

temp1 = pow(((double)(x - c) / (double)sig), (double)2.0);

temp2 = exp(-0.5*temp1)*pow((double)2, (double)frac);

return temp2;


}



template <int BLOCK_SIZE>
__global__ void outmflayer(int32_t *Chromosom, int32_t *chromosomSC, int32_t* X, int32_t *Target, int32_t* rules, int64_t* error)
{

	

	int32_t *Y;
	int64_t *F, *outmf;

	int32_t *UX, *WX, *wbar;
	int64_t totw;
	int64_t out;


	totw = 0, out = 0;

	UX = new int32_t[mflayrs];
	memset(UX, 0, 4 * mflayrs);

	WX = new int32_t[nods];
	memset(WX, 0, 4 * nods);

	wbar = new int32_t[nods];
	memset(wbar, 0, 4 * nods);

	outmf = new int64_t[nods];
	memset(outmf, 0, 8 * nods);

	Y = new int32_t[data_len];
	memset(Y, 0, 4 * data_len);

	F = new int64_t[nods];
	memset(F, 0, 8 * nods);




	int64_t temp;
	int64_t temp2;
	temp = 0, temp2 = 0;






	int data = blockDim.x * blockIdx.x + threadIdx.x;

	if (data < data_len) {
		//for (int data = 0; data < data_len; ++data) {



		for (int i = 0; i < 4; ++i) {
			__syncthreads();
			for (int j = 0; j < 2; j++) {

				UX[i*number_of_mffunc + j] = gussM_cuda(X[data*number_of_inputs + i], chromosomSC[j + i*number_of_mffunc + number_of_inputs*number_of_mffunc], chromosomSC[j + i*number_of_mffunc]);

				__syncthreads();

			}
		}

		//cout << " UX " << endl;
		// for (int i = 0; i < number_of_inputs*number_of_mffunc; ++i) {

		//cout << UX[i]<<endl;
		//}

		//memcpy(F, L, 4 * nods);


		__syncthreads();
		for (int i = 0; i < nods; ++i) {
			__syncthreads();

			F[i] = Chromosom[i];
		}
		__syncthreads();
		for (int i = 1; i < number_of_inputs + 1; ++i) {
			__syncthreads();
			for (int j = 0; j < nods; ++j) {

				//temp = X[data*number_of_inputs + i - 1];

				//temp *= Chromosom[j + i*nods];
				//F[j] += (temp >> frac);
				mult_cuda(temp, X[data*number_of_inputs + i - 1], Chromosom[j + i*nods]);
				F[j] += temp;

			}
		}




		__syncthreads();
		for (int i = 0; i < nods; ++i) {
			WX[i] = one;
			__syncthreads();
			for (int j = 0; j < number_of_inputs; ++j) {
				int k;

				k = rules[i * number_of_inputs + j];

				//temp = WX[i];
				// temp *= UX[k];
				//WX[i] = temp >> frac;

				mult_cuda(WX[i], WX[i], UX[k]);


			}
			totw += WX[i];


		}

		__syncthreads();
		if (totw == 0)
			totw = one;

		//	cout << " WX" << endl;
		//for (int i = 0; i < nods; ++i) {

		//cout << WX[i]<<endl;
		//}


		__syncthreads();
		for (int i = 0; i < nods; ++i) {

			//temp2 = WX[i];
			//wbar[i] = (temp2 << frac) / totw;

			divide_cuda(wbar[i], WX[i], totw);

			//temp = wbar[i];

			//temp *= F[i];

			//outmf[i] = temp >> frac;

			mult_cuda(outmf[i], wbar[i], F[i]);
			out += outmf[i];




		}


		

		Y[data] = out;

		//temp = pow((Target[data] - Y[data]), 2);
		//temp = temp >> frac;
		//error += temp;
		pow2_cuda(error[data], (Target[data] - Y[data]));
		//error += temp;
		
		__syncthreads();
		//error[data] = F[1];

		memset(F, 0, 8 * nods);
		memset(WX, 0, 4 * nods);
		memset(wbar, 0, 4 * nods);
		memset(outmf, 0, 8 * nods);
		out = 0;
		totw = 0;
		temp = 0;
		temp2 = 0;
		//X = X + number_of_inputs;

	}


	delete[] UX;
	delete[] WX;
	delete[] wbar;
	delete[] outmf;
	delete[] F;
	delete[] Y;


}


extern "C"
int32_t RMSE_cal(int32_t *Chromosom, int32_t *chromosomSC) {



	if (abs(chromosomSC[h_number_of_inputs*h_number_of_mffunc]) < 4096) {
		return 2147483646;
	}
	if (abs(chromosomSC[1 + h_number_of_inputs*h_number_of_mffunc]) < 4096) {
		return 2147483646;
	}
	if (abs(chromosomSC[2 + h_number_of_inputs*h_number_of_mffunc]) < 4096) {
		return 2147483646;
	}
	if (abs(chromosomSC[3 + h_number_of_inputs*h_number_of_mffunc]) < 4096) {
		return 2147483646;
	}

	if (abs(chromosomSC[4 + h_number_of_inputs*h_number_of_mffunc]) < 4096) {
		return 2147483646;
	}
	if (abs(chromosomSC[5 + h_number_of_inputs*h_number_of_mffunc]) < 4096) {
		return 2147483646;
	}

	if (abs(chromosomSC[6 + h_number_of_inputs*h_number_of_mffunc]) < 4096) {
		return 2147483646;
	}


	if (abs(chromosomSC[7 + h_number_of_inputs*h_number_of_mffunc]) < 4096) {
		return 2147483646;
	}

	/*
	if (abs(chromosomSC[8 + number_of_inputs*number_of_mffunc]) < 4096) {
	return 2147483646;
	}
	if (abs(chromosomSC[9 + number_of_inputs*number_of_mffunc]) < 4096) {
	return 2147483646;
	}



	if (abs(chromosomSC[10 + number_of_inputs*number_of_mffunc]) < 4096) {
	return 2147483646;
	}
	if (abs(chromosomSC[11 + number_of_inputs*number_of_mffunc]) < 4096) {
	return 2147483646;
	}


	if (abs(chromosomSC[12 + number_of_inputs*number_of_mffunc]) < 4096) {
	return 2147483646;
	}
	if (abs(chromosomSC[13 + number_of_inputs*number_of_mffunc]) < 4096) {
	return 2147483646;
	}
	if (abs(chromosomSC[14 + number_of_inputs*number_of_mffunc]) < 4096) {
	return 2147483646;
	}
	if (abs(chromosomSC[15 + number_of_inputs*number_of_mffunc]) < 4096) {
	return 2147483646;
	}
	*/

	size_t f, t;
	cudaMemGetInfo(&f, &t); 
	fprintf(stdout, "Free: %d Total: %d\n", f / 1024, t / 1024);

	int32_t *d_Chromosom, *d_chromosomSC;

	size_t sizeSC = SCsize * sizeof(int32_t);
	if (cudaSuccess != cudaMalloc(&d_chromosomSC, sizeSC))
		printf("Error!\n");

	

	cudaMemGetInfo(&f, &t);
	fprintf(stdout, "Free: %d Total: %d\n", f / 1024, t / 1024);

	size_t sizeL = tempsize * sizeof(int32_t);
	

	if (cudaSuccess != cudaMalloc(&d_Chromosom, sizeL))
		printf("Error!\n");

	cudaMemGetInfo(&f, &t);
	fprintf(stdout, "Free: %d Total: %d\n", f / 1024, t / 1024);
	
	int32_t N, total = 0;
	int64_t* error;
	error = new int64_t[h_data_len];

	size_t sizeX = h_number_of_inputs* h_data_len * sizeof(int32_t);
	int32_t* d_x;
	

	if (cudaSuccess != cudaMalloc(&d_x, sizeX))
		printf("Error!\n");

	cudaMemGetInfo(&f, &t);
	fprintf(stdout, "Free: %d Total: %d\n", f / 1024, t / 1024);

	size_t sizeY =  h_data_len * sizeof(int32_t);
	int32_t* d_Target;
	

	if (cudaSuccess != cudaMalloc(&d_Target, sizeY))
		printf("Error!\n");

	cudaMemGetInfo(&f, &t);
	fprintf(stdout, "Free: %d Total: %d\n", f / 1024, t / 1024);

	size_t sizeE = h_data_len * sizeof(int64_t);
	int64_t* d_e;
	


	if (cudaSuccess != cudaMalloc(&d_e, sizeE))
		printf("Error!\n");

	cudaMemGetInfo(&f, &t);
	fprintf(stdout, "Free: %d Total: %d\n", f / 1024, t / 1024);


	int32_t* rule_list_cuda;
	size_t sizeRule = h_nods * h_number_of_inputs * sizeof(int32_t);

	

	if (cudaSuccess != cudaMalloc(&rule_list_cuda, sizeRule))
		printf("Error!\n");

	cudaMemGetInfo(&f, &t);
	fprintf(stdout, "Free: %d Total: %d\n", f / 1024, t / 1024);

 	cudaMemcpy(rule_list_cuda, rule_list_host, sizeRule, cudaMemcpyHostToDevice);

	cudaMemcpy(d_x, inputs1, sizeX, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Target, output1, sizeY, cudaMemcpyHostToDevice);

	cudaMemcpy(d_chromosomSC, chromosomSC, sizeSC, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Chromosom, Chromosom, sizeL, cudaMemcpyHostToDevice);


	int threadsPerBlock = 170;
	int blocksPerGrid = 2;

	fprintf(stdout, "before kernel\n");
	outmflayer <16> <<<blocksPerGrid, threadsPerBlock >>> (d_Chromosom, d_chromosomSC, d_x, d_Target, rule_list_cuda, d_e);
	fprintf(stdout, "after kernel\n");
	if (cudaSuccess != cudaGetLastError())
		printf("Error!\n");

	// Copy result from device memory to host memory
	// h_C contains the result in host memory
	cudaMemcpy(error, d_e, sizeE, cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(d_x);
	cudaFree(d_e);
	cudaFree(rule_list_cuda);
	cudaFree(d_Chromosom);
	cudaFree(d_chromosomSC);
	cudaFree(d_Target);
	
	//pow2(temp, (Target[data] - Y[data]));
	//total = 0;
	for (int i = 0;i < h_data_len;++i) {
		total +=error[i] ;
	}
	return (fixsqr_cuda((total / h_data_len)));
}



