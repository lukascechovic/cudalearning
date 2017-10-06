#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void test_matrixvectmulti(void);

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(float *a, float *b, float *c, int n)
{
    // Get our global thread ID
    //blockDim pocet threadov v jednom bloku
    //threadIdx 0 az max pocet, jeden thread obsluhovane jednym jadrom idealne
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    // n vector size
    if (id < n)
        c[id] = a[id] + b[id];
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void vectmatrixMulti(float *matrix, float *vect, float *result, int M, int N)
{
    // Get our global thread ID
    //blockDim pocet threadov v jednom bloku
    //threadIdx 0 az max pocet, jeden thread obsluhovane jednym jadrom idealne
    //global thread ID
    //v threade pocitam
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    // n vector size
    if (id < N)
    {
      float sum = 0;
      for (int i=0; i<M; i++)
      {
        sum+= matrix[(id*M)+i] * vect[i];
      }
      result[id] = sum;
    }

}

void matrixvectorMulti (void)
{
}

int main( int argc, char* argv[] )
{
    //test_vectadd();
    test_matrixvectmulti();

    return 0;
}

void test_vectadd(void)
{
  // Size of vectors
  int n = 100000;

  // Host input vectors
  float *h_a;
  float *h_b;
  //Host output vector
  float *h_c;

  // Device input vectors
  float *d_a;
  float *d_b;
  //Device output vector
  float *d_c;

  // Size, in bytes, of each vector
  size_t bytes = n*sizeof(float);

  // Allocate memory for each vector on host
  h_a = (float*)malloc(bytes);
  h_b = (float*)malloc(bytes);
  h_c = (float*)malloc(bytes);

  // Allocate memory for each vector on GPU
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  int i;
  // Initialize vectors on host
  for( i = 0; i < n; i++ ) {
      h_a[i] = sin(i)*sin(i);
      h_b[i] = cos(i)*cos(i);
  }

  // Copy host vectors to device
  cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);

  int blockSize, gridSize;

  // Number of threads in each thread block
  blockSize = 1024;

  // Number of thread blocks in grid
  gridSize = (int)ceil((float)n/blockSize);

  // Execute the kernel
  vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

  // ekvivalent sync pri usb linux
  cudaDeviceSynchronize();
  // Copy array back to host
  cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );

  // Sum up vector c and print result divided by n, this should equal 1 within error
  float sum = 0;
  for(i=0; i<n; i++)
      sum += h_c[i];
  printf("final result: %f\n", sum/n);

  // Release device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // Release host memory
  free(h_a);
  free(h_b);
  free(h_c);
}

void test_matrixvectmulti(void)
{
  // Size of vectors
  //pocet riadkov rows
  int n = 1000;

  //pocet stlpcov collumns
  //pocet prvkov v riadku
  int m = 100;

  // Host input vectors
  float *h_a;
  float *h_b;
  //Host output vector
  float *h_c;

  // Device input vectors
  float *d_a;
  float *d_b;
  //Device output vector
  float *d_c;

  size_t input_size = m*sizeof(float);
  size_t output_size = n*sizeof(float);
  size_t matrix_size = m*n*sizeof(float);

  // Allocate memory for each vector on host
  //matrix
  h_a = (float*)malloc(matrix_size);
  //vector
  h_b = (float*)malloc(input_size);
  //result
  h_c = (float*)malloc(output_size);

  // Allocate memory for each vector on GPU
  cudaMalloc(&d_a, matrix_size);
  cudaMalloc(&d_b, input_size);
  cudaMalloc(&d_c, output_size);

  // Initialize vectors on host
  for(int i = 0; i < n; i++ )
  {
      for (int j = 0; j < m; j++)
      {
        h_a[(i*m)+j] = cos(j)*cos(j);
      }
  }

  for(int i = 0; i < m; i++ )
  {
      h_b[i] = sin(i)*sin(i);
  }

  // Copy host vectors to device
  cudaMemcpy( d_a, h_a, matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy( d_b, h_b, input_size, cudaMemcpyHostToDevice);

  int blockSize, gridSize;

  // Number of threads in each thread block
  blockSize = 1024;

  // Number of thread blocks in grid
  gridSize = (int)ceil((float)n/blockSize);

  // Execute the kernel
  vectmatrixMulti<<<gridSize, blockSize>>>(d_a, d_b, d_c, m, n);

  // ekvivalent sync pri usb linux
  cudaDeviceSynchronize();
  // Copy array back to host
  cudaMemcpy( h_c, d_c, output_size, cudaMemcpyDeviceToHost );

  // Sum up vector c and print result divided by n, this should equal 1 within error
  float sum = 0;
  for(unsigned int i=0; i<n; i++)
      sum += h_c[i];
  printf("final result: %f\n", sum/n);

  // Release device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // Release host memory
  free(h_a);
  free(h_b);
  free(h_c);
}
