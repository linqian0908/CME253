#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "headers.h"

/* macro to index a 1D memory array with 2D indices in column-major order */
#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )
// GPU macro
#define THREADS_PER_BLOCK 32
typedef float floatT;

__global__ void gpu_eh(const int size, const int x, const floatT t, const floatT sigma,
	const int idx, const int idy, const int k,  floatT *e, floatT *hx, floatT *hy) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	__shared__ floatT s_e[THREADS_PER_BLOCK+1][THREADS_PER_BLOCK+1];
	__shared__ floatT s_hx[THREADS_PER_BLOCK+1][THREADS_PER_BLOCK+2];
	__shared__ floatT s_hy[THREADS_PER_BLOCK+2][THREADS_PER_BLOCK+1];
	
	if (i>=(size-1) || j>=(size-1)) { 
		s_e[threadIdx.x][threadIdx.y] = 0.0;
		s_hx[threadIdx.x][threadIdx.y+1] = 0.0;
		s_hy[threadIdx.x+1][threadIdx.y] = 0.0; 
	}
	
	// read interior data for each threadblock from global memory
	else {
		s_e[threadIdx.x][threadIdx.y] = e[INDX(i,j,size)];
		s_hx[threadIdx.x][threadIdx.y+1] = hx[INDX(i,j,size)];
		s_hy[threadIdx.x+1][threadIdx.y] = hy[INDX(i,j,size-1)];
	}
	
	// read boundary data for each threadblock necesary for updating interior data
	if (threadIdx.x==(THREADS_PER_BLOCK-1)) { //top
		if (i<(size-2)) {
			s_e[THREADS_PER_BLOCK][threadIdx.y] = e[INDX(i+1,j,size)];
			s_hx[THREADS_PER_BLOCK][threadIdx.y+1] = hx[INDX(i+1,j,size)];
			s_hy[THREADS_PER_BLOCK+1][threadIdx.y] = hy[INDX(i+1,j,size-1)];
		}
		else {
			s_e[THREADS_PER_BLOCK][threadIdx.y] = 0.0;
			s_hx[THREADS_PER_BLOCK][threadIdx.y+1] = 0.0;
			s_hy[THREADS_PER_BLOCK+1][threadIdx.y] = 0.0;
		}
	}
	if (threadIdx.y==(THREADS_PER_BLOCK-1)) { //right
		if (j<(size-2)) {
			s_e[threadIdx.x][THREADS_PER_BLOCK] = e[INDX(i,j+1,size)];
			s_hx[threadIdx.x][THREADS_PER_BLOCK+1] = hx[INDX(i,j+1,size)];
			s_hy[threadIdx.x+1][THREADS_PER_BLOCK] = hy[INDX(i,j+1,size-1)];
		}
		else {
			s_e[threadIdx.x][THREADS_PER_BLOCK] = 0.0;
			s_hx[threadIdx.x][THREADS_PER_BLOCK+1] = 0.0;
			s_hy[threadIdx.x+1][THREADS_PER_BLOCK] = 0.0;
		}
	}
	if (threadIdx.y==0) { // left
		if (j>0) {
			s_hx[threadIdx.x][0] = hx[INDX(i,j-1,size)];
		}
		else {
			s_hx[threadIdx.x][0] = 0.0;
		}
	}
	if (threadIdx.x==0) { // bottom
		if (i>0) {
			s_hy[0][threadIdx.y] = hy[INDX(i-1,j,size-1)];
		}
		else {
			s_hy[0][threadIdx.y] = 0.0;
		}
	}
	__syncthreads();
	
	if (i>=(size-1) || j>=(size-1)) { return; }
	
	// compute update in shared memory
	if (i>0 && j>0) {
		s_e[threadIdx.x][threadIdx.y] += (s_hy[threadIdx.x+1][threadIdx.y]-s_hy[threadIdx.x][threadIdx.y]) 
						- (s_hx[threadIdx.x][threadIdx.y+1]-s_hx[threadIdx.x][threadIdx.y]);
		if (threadIdx.x==(THREADS_PER_BLOCK-1)) {
			s_e[THREADS_PER_BLOCK][threadIdx.y] += (s_hy[THREADS_PER_BLOCK+1][threadIdx.y]-s_hy[THREADS_PER_BLOCK][threadIdx.y])
						- (s_hx[THREADS_PER_BLOCK][threadIdx.y+1]-s_hx[THREADS_PER_BLOCK][threadIdx.y]);
		}			
		if (threadIdx.y==(THREADS_PER_BLOCK-1)) {
			s_e[threadIdx.x][THREADS_PER_BLOCK] += (s_hy[threadIdx.x+1][THREADS_PER_BLOCK]-s_hy[threadIdx.x][THREADS_PER_BLOCK])
						- (s_hx[threadIdx.x][THREADS_PER_BLOCK+1]-s_hx[threadIdx.x][THREADS_PER_BLOCK]);
		}		
		if (i==idx && j==idy) { s_e[threadIdx.x][threadIdx.y] -= FJ(k, x, t, sigma); }
	}
	__syncthreads();
	s_hx[threadIdx.x][threadIdx.y+1] -= 0.5*(s_e[threadIdx.x][threadIdx.y+1] - s_e[threadIdx.x][threadIdx.y]);
	s_hy[threadIdx.x+1][threadIdx.y] += 0.5*(s_e[threadIdx.x+1][threadIdx.y] - s_e[threadIdx.x][threadIdx.y]);
	
	// writing shared memory out
	if (i>0 && j>0) {
		e[INDX(i,j,size)] = s_e[threadIdx.x][threadIdx.y];
	}
	if (i>0) {
		hx[INDX(i,j,size)] = s_hx[threadIdx.x][threadIdx.y+1];
	}
	if (j>0) {
		hy[INDX(i,j,size-1)] = s_hy[threadIdx.x+1][threadIdx.y];
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
	printf("fdtd_sync: GPU using share memory for both E and H (fdtd_smem).\n" );
	
	int dev;
	cudaDeviceProp deviceProp;
	checkCUDA( cudaGetDevice( &dev ) );
	checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
	printf("Using GPU %d: %s\n", dev, deviceProp.name );
	
	floatT L = 799.0; //1598.0;
	floatT hx = 1.0;
	floatT ht = hx/sqrt(2.0)/3;
 	floatT sigma = 200*ht;

	fprintf(stdout, "fj output is %f\n", FJ(500, hx, ht, sigma));

	int size = (int) L/hx+1;
	int idx = (int) (0.625*L/hx)+1;
	int idy = (int) (0.5*L/hx)+1;
	fprintf(stdout, "size is %d, source is at idx=%d and idy=%d.\n", size, idx, idy);

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
	host_fdtd(size, hx, ht, sigma, idx, idy, k_beg, k_end, h_E, h_Hx, h_Hy);
	/*
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
	*/
	t_end = clock();
	fprintf(stdout, "CPU calculation time for %d iteration is %f s\n", k_end, (float)(t_end - t_begin) / CLOCKS_PER_SEC);
	
	// GPU execution
	
	dim3 threads( THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
	dim3 blocks( ((size-1)/threads.x)+1, ((size-1)/threads.y)+1, 1);
	fprintf(stdout, "block size is %d by %d.\n", blocks.x, blocks.y);

	/* GPU timer */
	cudaEvent_t start, stop;
	checkCUDA( cudaEventCreate( &start ) );
    checkCUDA( cudaEventCreate( &stop ) );
	checkCUDA( cudaEventRecord( start, 0 ) );

	/* launch the kernel on the GPU */
	for (int k=k_beg; k<=k_end; k++) {
		gpu_eh<<< blocks, threads >>>( size, hx, ht, sigma, idx, idy, k, d_E, d_Hx, d_Hy );
		checkKERNEL();
	}
	
	/* stop the timers */
	checkCUDA( cudaEventRecord( stop, 0 ) );
	checkCUDA( cudaEventSynchronize( stop ) );
	float gpuTime;
	checkCUDA( cudaEventElapsedTime( &gpuTime, start, stop ) );

	printf("GPU naive calculation time %f ms\n", gpuTime );
	
	floatT *out_E, *out_Hx, *out_Hy;
	out_E = (floatT *) malloc (numbytes_E);
	out_Hx = (floatT *) malloc (numbytes_H);
	out_Hy = (floatT *) malloc (numbytes_H);

	checkCUDA( cudaMemcpy( out_E, d_E, numbytes_E, cudaMemcpyDeviceToHost ) );
	checkCUDA( cudaMemcpy( out_Hx, d_Hx, numbytes_H, cudaMemcpyDeviceToHost ) );
	checkCUDA( cudaMemcpy( out_Hy, d_Hy, numbytes_H, cudaMemcpyDeviceToHost ) );
	/*
	for( int i = 0; i < size; i++ )	{
		for ( int j = 0; j<size; j++ ) {
			if (abs(h_E[INDX(i,j,size)])>0 || abs(out_E[INDX(i,j,size)])>0) {
				printf("E element %d, %d: CPU %e vs GPU %e\n",i,j,h_E[INDX(i,j,size)],out_E[INDX(i,j,size)] );
			}
		}
	}
	
	for( int i = 0; i < size; i++ )	{
		for ( int j = 0; j<size-1; j++ ) {
			if (abs(h_Hx[INDX(i,j,size)])>0 || abs(out_Hx[INDX(i,j,size)])>0) {
				printf("Hx element %d, %d: CPU %e vs GPU %e\n",i,j,h_Hx[INDX(i,j,size)],out_Hx[INDX(i,j,size)] );
			}
		}
	} 
	
	for( int i = 0; i < size-1; i++ ) {
		for ( int j = 0; j<size; j++ ) {
			if (abs(h_Hy[INDX(i,j,size-1)])>0 || abs(out_Hy[INDX(i,j,size-1)])>0) {
				printf("Hy element %d, %d: CPU %e vs GPU %e\n",i,j,h_Hy[INDX(i,j,size-1)],out_Hy[INDX(i,j,size-1)] );
			}
		}
	} 
	*/
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
