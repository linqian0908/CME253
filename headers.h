#define checkCUDA(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);}

#define checkKERNEL()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} \
  if( (cudaDeviceSynchronize()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);}
   
#define FJ(n, hx, ht, sigma) 1000*(exp(-pow(((n-0.5)*ht/sigma-4),2))*sin(2*M_PI*(n-0.5)*hx/800/sqrt(2.0)))

