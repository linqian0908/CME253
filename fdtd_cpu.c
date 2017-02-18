#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


/* macro to index a 1D memory array with 2D indices in column-major order */
#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )
#define FJ(n, hx, ht, sigma) 1000*(exp(-pow(((n-0.5)*ht/sigma-4),2))*sin(2*M_PI*(n-0.5)*hx/800/sqrt(2.0)))
typedef float floatT;

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
	
	floatT L = 799.0;
	floatT hx = 1.0;
	floatT ht = hx/sqrt(2.0)/3;
 	floatT sigma = 200*ht;

	fprintf(stdout, "fj output is %f\n", FJ(500, hx, ht, sigma));

	int size = (int) L/hx+1;
	int idx = (int) (0.625*L/hx)+1;
	int idy = (int) (0.5*L/hx)+1;

	floatT *h_E, *h_Hx, *h_Hy;

	size_t num_E = size * size;
	size_t num_H = (size - 1)*size;	
	fprintf(stdout, "total memory allocated is %lu\n", (num_E+2*num_H)*sizeof(floatT));
	
	clock_t t_begin, t_end;	
	t_begin = clock();
	h_E = (floatT *) calloc (num_E, sizeof(floatT));
	h_Hx = (floatT *) calloc (num_H, sizeof(floatT));
	h_Hy = (floatT *) calloc (num_H, sizeof(floatT));

	h_E[INDX(idx, idy, size)] = - FJ(1, hx, ht, sigma);
	t_end = clock();
	fprintf(stdout, "CPU memory allocation time is %f s\n", (float)(t_end - t_begin) / CLOCKS_PER_SEC);
	
	int k_beg = 2;
	int k_end = 1500;
	
	t_begin = clock();
	host_fdtd(size, hx, ht, sigma, idx, idy, k_beg, k_end, h_E, h_Hx, h_Hy);
	t_end = clock();	
	fprintf(stdout, "CPU calculation time for %d iteration is %f s\n", k_end, (float)(t_end - t_begin) / CLOCKS_PER_SEC);

	FILE *fp;
	fp = fopen("/home/linqian/Desktop/2017Winter/CME253/project/code/cpu_E.f","wb");
	fwrite(h_E,sizeof(floatT),num_E,fp);
	fclose(fp);
	fprintf(stdout, "finish writing E.\n");
	
	fp = fopen("/home/linqian/Desktop/2017Winter/CME253/project/code/cpu_Hx.f","wb");
	fwrite(h_Hx,sizeof(floatT),num_H,fp);
	fclose(fp);
	fprintf(stdout, "finish writing Hx.\n");
	
	fp = fopen("/home/linqian/Desktop/2017Winter/CME253/project/code/cpu_Hy.f","wb");
	fwrite(h_Hy,sizeof(floatT),num_H,fp);
	fclose(fp);
	fprintf(stdout, "finish writing Hy.\n");
	
	free(h_E);
	free(h_Hx);
	free(h_Hy);	
	
	return 0;
}
