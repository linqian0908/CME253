#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "time.h"

/* macro to index a 1D memory array with 2D indices in column-major order */
#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )
#define FJ(n, hx, ht, sigma) (exp(-pow(((n-0.5)*ht/sigma-4),2))*sin(2*M_PI*(n-0.5)*hx/800/sqrt(2.0)))

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
// #define RADIUS 1

__global__ void gpu_fdtd_shmem(const int size, const int x, const double t, const double sigma,
    const int idx, const int idy, const double e_0,
	const int k_beg, const int k_end, double *e, double *hx, double *hy) {
	// printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);

    
	__shared__ double share_e[THREADS_PER_BLOCK_X + 2][THREADS_PER_BLOCK_Y + 2];
	__shared__ double share_hx[THREADS_PER_BLOCK_X + 2][THREADS_PER_BLOCK_Y + 2];
	__shared__ double share_hy[THREADS_PER_BLOCK_X + 2][THREADS_PER_BLOCK_Y + 2];

	int globalIndex_x = blockIdx.x * blockDim.x + threadIdx.x;
	int localIndex_x = threadIdx.x + 1;

	int globalIndex_y = blockIdx.y * blockDim.y + threadIdx.y;
	int localIndex_y = threadIdx.y + 1;

	for (int k = k_beg; k <= k_end; k++) {
	// int k = k_beg;

		share_e[localIndex_x][localIndex_y] = e[INDX(globalIndex_x, globalIndex_y, size)];
		share_hx[localIndex_x][localIndex_y] = hx[INDX(globalIndex_x, globalIndex_y, size)];
		share_hy[localIndex_x][localIndex_y] = hy[INDX(globalIndex_x, globalIndex_y, size-1)];

		if (threadIdx.x < 1 && globalIndex_x >= 1) {

			share_e[localIndex_x - 1][localIndex_y] = e[INDX(globalIndex_x - 1, globalIndex_y, size)];
			share_hx[localIndex_x - 1][localIndex_y] = hx[INDX(globalIndex_x - 1 , globalIndex_y, size)];
			share_hy[localIndex_x - 1][localIndex_y] = hy[INDX(globalIndex_x - 1 , globalIndex_y, size-1)];
			//printf("threadIdx.x: %d, globalIndex_x: %d, threadIdx.y: %d, globalIndex_y: %d, e: %e, share_e: %e\n", 
			//	threadIdx.x, globalIndex_x,threadIdx.y, globalIndex_y, e[INDX(globalIndex_x - 1, globalIndex_y, size)], share_e[localIndex_x - 1][localIndex_y]);
		}

		if (threadIdx.x < 1 && (globalIndex_x + THREADS_PER_BLOCK_X) < size){
		// if (threadIdx.x < 1 && globalIndex_x < (size - 1)){
			share_e[THREADS_PER_BLOCK_X + 1][localIndex_y] = e[INDX(globalIndex_x + THREADS_PER_BLOCK_X, globalIndex_y, size)];
			share_hx[THREADS_PER_BLOCK_X + 1][localIndex_y] = hx[INDX(globalIndex_x + THREADS_PER_BLOCK_X, globalIndex_y, size)];
			share_hy[THREADS_PER_BLOCK_X + 1][localIndex_y] = hy[INDX(globalIndex_x + THREADS_PER_BLOCK_X, globalIndex_y, size-1)];
		}

		if (threadIdx.y < 1 && globalIndex_y >= 1) {
			share_e[localIndex_x][localIndex_y - 1] = e[INDX(globalIndex_x, globalIndex_y - 1, size)];
			share_hx[localIndex_x][localIndex_y - 1] = hx[INDX(globalIndex_x, globalIndex_y - 1, size)];
			share_hy[localIndex_x][localIndex_y - 1] = hy[INDX(globalIndex_x, globalIndex_y - 1, size-1)];
		}

		if (threadIdx.y < 1 && (globalIndex_y + THREADS_PER_BLOCK_Y) < size) {
		// if (threadIdx.y < 1 && globalIndex_y < (size - 1)) {
			share_e[localIndex_x][THREADS_PER_BLOCK_Y + 1] = e[INDX(globalIndex_x, globalIndex_y + THREADS_PER_BLOCK_Y, size)];
			share_hx[localIndex_x][THREADS_PER_BLOCK_Y + 1] = hx[INDX(globalIndex_x, globalIndex_y + THREADS_PER_BLOCK_Y, size)];
			share_hy[localIndex_x][THREADS_PER_BLOCK_Y + 1] = hy[INDX(globalIndex_x, globalIndex_y + THREADS_PER_BLOCK_Y, size - 1)];
		}

		if (threadIdx.x < 1 && (globalIndex_x + THREADS_PER_BLOCK_X) < size && threadIdx.y < 1 && (globalIndex_y + THREADS_PER_BLOCK_Y) < size){
			share_e[THREADS_PER_BLOCK_X + 1][THREADS_PER_BLOCK_Y + 1] = e[INDX(globalIndex_x + THREADS_PER_BLOCK_X, globalIndex_y + THREADS_PER_BLOCK_Y, size)];
			share_hx[THREADS_PER_BLOCK_X + 1][THREADS_PER_BLOCK_Y + 1] = hx[INDX(globalIndex_x + THREADS_PER_BLOCK_X, globalIndex_y + THREADS_PER_BLOCK_Y, size)];
			share_hy[THREADS_PER_BLOCK_X + 1][THREADS_PER_BLOCK_Y + 1] = hy[INDX(globalIndex_x + THREADS_PER_BLOCK_X, globalIndex_y + THREADS_PER_BLOCK_Y, size-1)];
		}	

		if (threadIdx.x < 1 && globalIndex_x >= 1 && threadIdx.y < 1 && globalIndex_y >= 1) {
			share_e[0][0] = e[INDX(globalIndex_x - 1, globalIndex_y - 1, size)];
			share_hx[0][0] = hx[INDX(globalIndex_x - 1, globalIndex_y - 1, size)];
			share_hy[0][0] = hy[INDX(globalIndex_x - 1, globalIndex_y - 1, size)];
		}

		__syncthreads();

		if (globalIndex_x > 0 && globalIndex_x < (size-1) && globalIndex_y > 0 && globalIndex_y < (size-1)) {
			share_e[localIndex_x][localIndex_y] = share_e[localIndex_x][localIndex_y]
			 + (share_hy[localIndex_x][localIndex_y] - share_hy[localIndex_x-1][localIndex_y])
				- (share_hx[localIndex_x][localIndex_y] - share_hx[localIndex_x][localIndex_y - 1]);


			if ((globalIndex_x) == idx && (globalIndex_y) == idy) {
				printf ("globalIndex_x: %d, globalIndex_y: %d, share_e: %e\n", globalIndex_x, globalIndex_y, share_e[localIndex_x][localIndex_y]);
				share_e[localIndex_x][localIndex_y] -= t/e_0 * FJ(k, x, t, sigma);
				printf ("globalIndex_x: %d, globalIndex_y: %d, share_e: %e\n", globalIndex_x, globalIndex_y, share_e[localIndex_x][localIndex_y]);
				// e[INDX(globalIndex_x, globalIndex_y, size)] = share_e[localIndex_x][localIndex_y];	
			}



			if (threadIdx.x == (THREADS_PER_BLOCK_X - 1) && globalIndex_x < (size - 2)) {
				share_e[localIndex_x + 1][localIndex_y] = share_e[localIndex_x + 1][localIndex_y]
				 + (share_hy[localIndex_x + 1][localIndex_y] - share_hy[localIndex_x][localIndex_y])
					- (share_hx[localIndex_x + 1][localIndex_y] - share_hx[localIndex_x + 1][localIndex_y - 1]);

				if ((globalIndex_x + 1) == idx && (globalIndex_y) == idy) {
					share_e[localIndex_x + 1][localIndex_y] -= t/e_0 * FJ(k, x, t, sigma);
				}	
			}

				

			if (threadIdx.y == (THREADS_PER_BLOCK_Y - 1) && globalIndex_y < (size - 2)) {
				share_e[localIndex_x][localIndex_y + 1] = share_e[localIndex_x][localIndex_y + 1]
				 + (share_hy[localIndex_x][localIndex_y + 1] - share_hy[localIndex_x-1][localIndex_y + 1])
					- (share_hx[localIndex_x][localIndex_y + 1] - share_hx[localIndex_x][localIndex_y]);

				if ((globalIndex_x) == idx && (globalIndex_y + 1) == idy) {
					share_e[localIndex_x][localIndex_y + 1] -= t/e_0 * FJ(k, x, t, sigma);
				}		

			}

			if (threadIdx.x == (THREADS_PER_BLOCK_X - 1) && globalIndex_x < (size - 2) && threadIdx.y == (THREADS_PER_BLOCK_Y - 1) && globalIndex_y < (size - 2)) {
				share_e[localIndex_x + 1][localIndex_y + 1] = share_e[localIndex_x + 1][localIndex_y + 1]
				 + (share_hy[localIndex_x + 1][localIndex_y + 1] - share_hy[localIndex_x][localIndex_y + 1])
					- (share_hx[localIndex_x + 1][localIndex_y + 1] - share_hx[localIndex_x + 1][localIndex_y]);

				if ((globalIndex_x + 1) == idx && (globalIndex_y + 1) == idy) {
					share_e[localIndex_x + 1][localIndex_y + 1] -= t/e_0 * FJ(k, x, t, sigma);
				}	
			}




			e[INDX(globalIndex_x, globalIndex_y, size)] = share_e[localIndex_x][localIndex_y];
		}

		

		__syncthreads();

		if (globalIndex_x < (size - 1) && globalIndex_y < size) {
			hy[INDX(globalIndex_x, globalIndex_y, size - 1)] = share_hy[localIndex_x][localIndex_y]
			+ 0.5 * (share_e[localIndex_x + 1][localIndex_y] - share_e[localIndex_x][localIndex_y]);
		}

		if (globalIndex_x < size && globalIndex_y < (size - 1)) {
			hx[INDX(globalIndex_x, globalIndex_y, size)] = share_hx[localIndex_x][localIndex_y]
			- 0.5 * (share_e[localIndex_x][localIndex_y + 1] - share_e[localIndex_x][localIndex_y]);
		}

		__syncthreads();

	}
}



void host_fdtd(const int size, const int x, const double t, const double sigma,
    const int idx, const int idy, const double e_0,
	const int k_beg, const int k_end, double *e, double *hx, double *hy) {
	for (int k = k_beg; k <= k_end; k++) {
		for (int i = 1; i < (size-1); i++) {
			for (int j = 1; j < (size-1); j++) {
				e[INDX(i, j, size)] = e[INDX(i, j, size)] + (hy[INDX(i, j, (size-1))] - hy[INDX(i-1, j, (size-1))])
				- (hx[INDX(i, j, size)] - hx[INDX(i, j-1, size)]);
			}
		}
		e[INDX(idx, idy, size)] = e[INDX(idx, idy, size)] - t/e_0 * FJ(k, x, t, sigma);


		for (int i = 0; i < (size-1); i++) {
			for (int j = 0; j < size; j++) {
				hy[INDX(i,j,(size-1))] = hy[INDX(i,j,(size-1))] + 0.5 * (e[INDX(i+1, j, size)] - e[INDX(i, j, size)]);
			}
		}

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < (size-1); j++) {
				hx[INDX(i, j, size)] = hx[INDX(i, j, size)] - 0.5 * (e[INDX(i, j+1, size)] - e[INDX(i, j, size)]);
			}
		}
	}
}

int main(int argc, char *argv[]) {
	
	// int hx = 40;
	// double ht = (hx/sqrt(2.0)/(3 * pow(10, 8)));
 // 	double sigma = (2*pow(10, -5));

	// fprintf(stdout, "fj output is %f\n", FJ(500, hx, ht, sigma));

	// int size = 21;
	// int idx = 15; 
	// int idy = 10;
	// double e_0 = 8.85 * pow(10,-3);

	int hx = 40;
	double ht = (hx/sqrt(2.0)/(3 * pow(10, 8)));
 	double sigma = (2*pow(10, -5));
	int peak = int(sigma*4/ht);
	fprintf(stdout, "fj output is %f at %d th timestep\n", FJ(peak, hx, ht, sigma),peak);

	int size = int(800/hx)+1;
	int idx = int(600/hx); 
	int idy = int(400/hx);
	double e_0 = 8.85 * pow(10,-3);

	double *h_E, *h_Hx, *h_Hy;

	size_t numbytes_E = size * size * sizeof(double);
	size_t numbytes_H = (size - 1) * size * sizeof(double);

	h_E = (double *) malloc (numbytes_E);
	h_Hx = (double *) malloc (numbytes_H);
	h_Hy = (double *) malloc (numbytes_H);

	for (int i = 0; i < size * size; i++) {
		h_E[i] = 0.0;
	}
	h_E[INDX(idx, idy, size)] = - ht / e_0 * FJ(1, hx, ht, sigma);


	fprintf(stdout, "%e\n", - ht / e_0 * FJ(1, hx, ht, sigma));


	for (int i = 0; i < size * (size - 1); i++) {
		h_Hx[i] = 0.0;
		h_Hy[i] = 0.0;
	}
	
	// GPU memory allocation and initialization
	double *d_E, *d_Hx, *d_Hy, *d_E_out;
	cudaMalloc( (void **) &d_E, numbytes_E );
	cudaMalloc( (void **) &d_Hx, numbytes_H );
	cudaMalloc( (void **) &d_Hy, numbytes_H );
	cudaMemset(d_E, 0.0, numbytes_E);
	cudaMemset(d_Hx, 0.0, numbytes_H);
	cudaMemset(d_Hy, 0.0, numbytes_H);

	cudaMemcpy(d_E, h_E, numbytes_E, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Hx, h_Hx, numbytes_H, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Hy, h_Hy, numbytes_H, cudaMemcpyHostToDevice);

	cudaMalloc( (void **) &d_E_out, numbytes_E );
	cudaMemset(d_E_out, 0.0, numbytes_E);

	int k_beg = 2;
	int k_end = 3;
	// int k_end = floor(1.3 * pow(10, -4) / ht);
	fprintf(stdout, "k end is %d\n", k_end);
	clock_t begin = clock();
	
	host_fdtd(size, hx, ht, sigma, idx, idy, e_0, k_beg, k_end, h_E, h_Hx, h_Hy);
	
	clock_t end = clock();
	float cpuTime = (float)(end - begin) / CLOCKS_PER_SEC;
	fprintf(stdout, "CPU time for %d elements was %f ms\n", size*size, cpuTime);	
	
	

	dim3 threads( THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
	dim3 blocks( (size/threads.x)+1, (size/threads.x)+1, 1);

	/* GPU timer */
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );

	/* launch the kernel on the GPU */
	gpu_fdtd_shmem<<< blocks, threads >>>( size, hx, ht, sigma, idx, idy, e_0, k_beg, k_end, d_E, d_Hx, d_Hy );

	/* stop the timers */
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	float gpuTime;
	cudaEventElapsedTime( &gpuTime, start, stop );

	printf("GPU naive time for %d elements was %f ms\n", size*size, gpuTime );
	
	double *out_E, *out_Hx, *out_Hy;
	out_E = (double *) malloc (numbytes_E);
	out_Hx = (double *) malloc (numbytes_H);
	out_Hy = (double *) malloc (numbytes_H);

	cudaMemcpy( out_E, d_E, numbytes_E, cudaMemcpyDeviceToHost );
	cudaMemcpy( out_Hx, d_Hx, numbytes_H, cudaMemcpyDeviceToHost );
	cudaMemcpy( out_Hy, d_Hy, numbytes_H, cudaMemcpyDeviceToHost );

	double th = 10E-12;
	for( int i = 0; i < size - 1; i++ )	{
		for ( int j = 0; j<size - 1; j++ ) {
			if (abs(h_E[INDX(i,j,size)]) > th || abs(out_E[INDX(i,j,size)]) > th || abs(h_Hx[INDX(i,j,size)]) > th || abs(out_Hx[INDX(i,j,size)]) > th || abs(h_Hy[INDX(i,j,size)]) >th || abs(out_Hy[INDX(i,j,size)]) > th) {
				printf("E: %d, %d: CPU %e vs GPU %e\n",i,j,h_E[INDX(i,j,size)],out_E[INDX(i,j,size)] );
				printf("Hx: %d, %d: CPU %e vs GPU %e\n",i,j,h_Hx[INDX(i,j,size)],out_Hx[INDX(i,j,size)] );
				printf("Hy: %d, %d: CPU %e vs GPU %e\n",i,j,h_Hy[INDX(i,j,size)],out_Hy[INDX(i,j,size)] );
			}
			
		}
	}

	int success = 1;
	double diff, thresh=1e-14;
	for( int i = 0; i < size; i++ )	{
		for ( int j = 0; j<size; j++ ) {
			diff = h_E[INDX(i,j,size)]- out_E[INDX(i,j,size)];
			if ( diff>thresh || diff<-thresh ) {
				printf("error in element %d, %d: CPU %f vs GPU %f\n",i,j,h_E[INDX(i,j,size)],out_E[INDX(i,j,size)] );
				success = 0;
				break;
			} /* end if */
		}
	} /* end for */

	if( success == 1 ) printf("PASS\n");
	else               printf("FAIL\n");

	free(h_E);
	free(h_Hx);
	free(h_Hy);	
	free(out_E);	
	free(out_Hx);
	free(out_Hy);
	cudaFree( d_E );
	cudaFree( d_Hx );
	cudaFree( d_Hy );

	cudaDeviceSynchronize();
	
	return 0;
}
