#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "headers.h"

/* macro to index a 1D memory array with 2D indices in column-major order */
#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )
// GPU macro
#define THREADS_PER_BLOCK 1024
#define N_BLOCKS 1
typedef float floatT;

__device__ int g_mutex;

__device__ void __gpu_sync() {
	int a;
	if (threadIdx.x==0) {
		atomicAdd(&g_mutex,1);
		while (g_mutex!=N_BLOCKS) { a=N_BLOCKS; }
	}
	__syncthreads();
}

__global__ void gpu_eh(const int size, const int x, const floatT t, const floatT sigma,
	const int idx, const int idy, const int k_beg, int k_end,  floatT *e, floatT *hx, floatT *hy) {
	__shared__ char dummy[40000];
	
	int i_base = threadIdx.x;
	int chunk = (size-1)/gridDim.x+1;
	int j_base = blockIdx.x*chunk;
	
	for (int k=k_beg; k<=k_end; k+=1) {
	
		for (int j = j_base; j<min(size-1,j_base+chunk); j+=1) {
			for (int i = i_base; i<(size-1); i+=blockDim.x) {
				if (i>0 && j>0) {
					e[INDX(i,j,size)] += (hy[INDX(i,j,size-1)]-hy[INDX(i-1,j,size-1)])- (hx[INDX(i,j,size)]-hx[INDX(i,j-1,size)]);
					if (i==idx && j==idy) { e[INDX(i,j,size)] -= FJ(k, x, t, sigma); }
				}
			}
		}
		
		for (int j = j_base; j<min(size-1,j_base+chunk); j+=1) {
			for (int i = i_base; i<(size-1); i+=blockDim.x) {
				hy[INDX(i,j,size-1)] += 0.5*(e[INDX(i+1,j,size)]-e[INDX(i,j,size)]);
				hx[INDX(i, j, size)] -= 0.5*(e[INDX(i, j+1, size)] - e[INDX(i, j, size)]);
			}
		}
		__gpu_sync();
	}
}

void host_fdtd(const int size, const int x, const floatT t, const floatT sigma,
    const int idx, const int idy, const int k_beg, const int k_end, 
    floatT *e, floatT *hx, floatT *hy) {
	for (int k = k_beg; k <= k_end; k++) {
		for (int i = 1; i < (size-1); i++) {
			for (int j = 1; j < (size-1); j++) {
				e[INDX(i, j, size)] += (hy[INDX(i, j, (size-1))] - hy[INDX(i-1, j, (size-1))])
				- (hx[INDX(i, j, size)] - hx[INDX(i, j-1, size)]);
			}
		}
		e[INDX(idx, idy, size)] -= FJ(k, x, t, sigma);


		for (int i = 0; i < (size-1); i++) {
			for (int j = 0; j < size; j++) {
				hy[INDX(i,j,(size-1))] += 0.5 * (e[INDX(i+1, j, size)] - e[INDX(i, j, size)]);
			}
		}

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < (size-1); j++) {
				hx[INDX(i, j, size)] -= 0.5 * (e[INDX(i, j+1, size)] - e[INDX(i, j, size)]);
			}
		}
	}
}

int main(int argc, char *argv[]) {
	printf("fdtd_sync: GPU using implicit CPU sync\n" );
	
	int dev;
	cudaDeviceProp deviceProp;
	checkCUDA( cudaGetDevice( &dev ) );
	checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
	printf("Using GPU %d: %s\n", dev, deviceProp.name );
	
	floatT L = 1598.0;
	floatT hx = 1.0;
	floatT ht = hx/sqrt(2.0)/3;
 	floatT sigma = 200*ht;

	fprintf(stdout, "fj output is %f\n", FJ(500, hx, ht, sigma));

	int size = (int) L/hx+1;
	int idx = (int) (0.625*L/hx)+1;
	int idy = (int) (0.5*L/hx)+1;
	fprintf(stdout, "size if %d, source is at idx=%d and idy=%d.\n", size, idx, idy);

	floatT *h_E, *h_Hx, *h_Hy;

	size_t num_E = size * size;
	size_t num_H = (size - 1)*size;	
	size_t numbytes_E = num_E*sizeof(floatT);
	size_t numbytes_H = num_H*sizeof(floatT);
		
	fprintf(stdout, "total memory allocated is %lu\n", numbytes_E+2*numbytes_H);
	
	clock_t t_begin, t_end;	
	t_begin = clock();
	h_E = (floatT *) calloc (num_E, sizeof(floatT));
	h_Hx = (floatT *) calloc (num_H, sizeof(floatT));
	h_Hy = (floatT *) calloc (num_H, sizeof(floatT));

	h_E[INDX(idx, idy, size)] = - FJ(1, hx, ht, sigma);
	
	// GPU memory allocation and initialization
	floatT *d_E, *d_Hx, *d_Hy;
	checkCUDA( cudaMalloc( (void **) &d_E, numbytes_E ) );
	checkCUDA( cudaMalloc( (void **) &d_Hx, numbytes_H ) );
	checkCUDA( cudaMalloc( (void **) &d_Hy, numbytes_H ) );

	checkCUDA( cudaMemcpy(d_E, h_E, numbytes_E, cudaMemcpyHostToDevice) );
	checkCUDA( cudaMemset(d_Hx, 0, numbytes_H) );
	checkCUDA( cudaMemset(d_Hy, 0, numbytes_H) );
	
	t_end = clock();
	fprintf(stdout, "Memory allocation time is %f s\n", (float)(t_end - t_begin) / CLOCKS_PER_SEC);

	int k_beg = 2;
	int k_end = 1500;
	
	t_begin = clock();
	//host_fdtd(size, hx, ht, sigma, idx, idy, k_beg, k_end, h_E, h_Hx, h_Hy);
	
	FILE *fp;
	fp = fopen("./cpu_E.f","rb");
	fread(h_E,sizeof(floatT),num_E,fp);
	fclose(fp);
	fprintf(stdout, "finish reading E.\n");
	
	fp = fopen("./cpu_Hx.f","rb");
	fread(h_Hx,sizeof(floatT),num_H,fp);
	fclose(fp);
	fprintf(stdout, "finish reading Hx.\n");
	
	fp = fopen("./cpu_Hy.f","rb");
	fread(h_Hy,sizeof(floatT),num_H,fp);
	fclose(fp);
	fprintf(stdout, "finish reading Hy.\n");
	
	t_end = clock();
	fprintf(stdout, "CPU calculation time for %d iteration is %f s\n", k_end, (float)(t_end - t_begin) / CLOCKS_PER_SEC);
	
	// GPU execution
	int h_mutex = 0;
	checkCUDA( cudaMemcpyToSymbol(g_mutex, &h_mutex,sizeof(int),0,cudaMemcpyHostToDevice) );
	
	dim3 threads( min(THREADS_PER_BLOCK,size), 1, 1);
	dim3 blocks( N_BLOCKS, 1, 1);
	fprintf(stdout, "block size is %d by %d.\n", blocks.x, blocks.y);

	/* GPU timer */
	cudaEvent_t start, stop;
	checkCUDA( cudaEventCreate( &start ) );
    checkCUDA( cudaEventCreate( &stop ) );
	checkCUDA( cudaEventRecord( start, 0 ) );

	/* launch the kernel on the GPU */

	gpu_eh<<< blocks, threads >>>( size, hx, ht, sigma, idx, idy, k_beg, k_end, d_E, d_Hx, d_Hy );
	checkKERNEL();

	
	/* stop the timers */
	checkCUDA( cudaEventRecord( stop, 0 ) );
	checkCUDA( cudaEventSynchronize( stop ) );
	float gpuTime;
	checkCUDA( cudaEventElapsedTime( &gpuTime, start, stop ) );

	fprintf(stdout, "GPU calculation time %f ms\n", gpuTime );
	checkCUDA( cudaMemcpyFromSymbol(&h_mutex, g_mutex, sizeof(int),0, cudaMemcpyDeviceToHost) );
	fprintf(stdout, "mutex value %d\n", h_mutex);

	floatT *out_E, *out_Hx, *out_Hy;
	out_E = (floatT *) malloc (numbytes_E);
	out_Hx = (floatT *) malloc (numbytes_H);
	out_Hy = (floatT *) malloc (numbytes_H);

	checkCUDA( cudaMemcpy( out_E, d_E, numbytes_E, cudaMemcpyDeviceToHost ) );
	checkCUDA( cudaMemcpy( out_Hx, d_Hx, numbytes_H, cudaMemcpyDeviceToHost ) );
	checkCUDA( cudaMemcpy( out_Hy, d_Hy, numbytes_H, cudaMemcpyDeviceToHost ) );

	int success = 1;
	floatT diff, thresh=1e-6;
	for( int i = 0; i < size; i++ )	{
		for ( int j = 0; j<size; j++ ) {
			diff = abs(1.0-out_E[INDX(i,j,size)]/h_E[INDX(i,j,size)]);
			if ( diff>thresh ) {
				printf("error in E element %d, %d: CPU %e vs GPU %e\n",i,j,h_E[INDX(i,j,size)],out_E[INDX(i,j,size)] );
				success = 0;
				break;
			}
		}
	} 

	for( int i = 0; i < size; i++ )	{
		for ( int j = 0; j<size-1; j++ ) {
			diff = abs(1.0-out_Hx[INDX(i,j,size)]/h_Hx[INDX(i,j,size)]);
			if ( diff>thresh ) {
				printf("error in Hx element %d, %d: CPU %e vs GPU %e\n",i,j,h_Hx[INDX(i,j,size)],out_Hx[INDX(i,j,size)] );
				success = 0;
				break;
			} 
		}
	} 
	
	for( int i = 0; i < size-1; i++ )	{
		for ( int j = 0; j<size; j++ ) {
			diff = abs(1.0-out_Hy[INDX(i,j,size-1)]/h_Hy[INDX(i,j,size-1)]);
			if ( diff>thresh) {
				printf("error in Hy element %d, %d: CPU %e vs GPU %e\n",i,j,h_Hy[INDX(i,j,size-1)],out_Hy[INDX(i,j,size-1)] );
				success = 0;
				break;
			} 
		}
	} 

	
	if( success == 1 ) printf("PASS\n");
	else               printf("FAIL\n");

	free(h_E);
	free(h_Hx);
	free(h_Hy);	
	free(out_E);
	free(out_Hx);
	free(out_Hy);
	checkCUDA( cudaFree( d_E ) );
	checkCUDA( cudaFree( d_Hx ) );
	checkCUDA( cudaFree( d_Hy ) );

	checkCUDA( cudaDeviceSynchronize() );
	
	return 0;
}
