#define checkCUDA(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);}

#define checkKERNEL()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} \
  if( (cudaDeviceSynchronize()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);}
   
/* macro to index a 1D memory array with 2D indices in column-major order */
#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )
#define FJ(n, hx, ht, sigma) 1000*(exp(-pow(((n-0.5)*ht/sigma-4),2))*sin(2*M_PI*(n-0.5)*hx/800/sqrt(2.0)))

// GPU macro
#define THREADS_PER_BLOCK 256
typedef float floatT;

