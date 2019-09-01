/*
 *  This file is part of RawTherapee.
 *
 *  Copyright (c) 2004-2010 Gabor Horvath <hgabor@rawtherapee.com>
 *
 *  RawTherapee is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  RawTherapee is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with RawTherapee.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "gauss.h"
#include <cmath>
#include <cstdlib>
#include "opthelper.h"
#include "jaggedarray.h"
#include "boxblur.h"
#include "stdio.h"
#include <time.h>
//#include "StopWatch.h"
#include <CL/cl.h>
#define MAX_SOURCE_SIZE (0x100000)

namespace
{

template<class T> void calculateYvVFactors( const T sigma, T &b1, T &b2, T &b3, T &B, T M[3][3])
{
    // coefficient calculation
    T q;

    if (sigma < 2.5) {
        q = 3.97156 - 4.14554 * sqrt (1.0 - 0.26891 * sigma);
    } else {
        q = 0.98711 * sigma - 0.96330;
    }

    T b0 = 1.57825 + 2.44413 * q + 1.4281 * q * q + 0.422205 * q * q * q;
    b1 = 2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q;
    b2 = -1.4281 * q * q - 1.26661 * q * q * q;
    b3 = 0.422205 * q * q * q;
    B = 1.0 - (b1 + b2 + b3) / b0;

    b1 /= b0;
    b2 /= b0;
    b3 /= b0;

    // From: Bill Triggs, Michael Sdika: Boundary Conditions for Young-van Vliet Recursive Filtering
    M[0][0] = -b3 * b1 + 1.0 - b3 * b3 - b2;
    M[0][1] = (b3 + b1) * (b2 + b3 * b1);
    M[0][2] = b3 * (b1 + b3 * b2);
    M[1][0] = b1 + b3 * b2;
    M[1][1] = -(b2 - 1.0) * (b2 + b3 * b1);
    M[1][2] = -(b3 * b1 + b3 * b3 + b2 - 1.0) * b3;
    M[2][0] = b3 * b1 + b2 + b1 * b1 - b2 * b2;
    M[2][1] = b1 * b2 + b3 * b2 * b2 - b1 * b3 * b3 - b3 * b3 * b3 - b3 * b2 + b3;
    M[2][2] = b3 * (b1 + b3 * b2);

}

// classical filtering if the support window is small and src != dst - OpenCL
  template<class T> void OpenCLgauss3x3 (OpenCL_helper* helper, T** RESTRICT src, T** RESTRICT dst, const int W, const int H, const T c0, const T c1, const T c2, const T b0, const T b1)
 {
    dst[0][0] = src[0][0];

     // do the first row on CPU, OpenMP if supported

    #ifdef _OPENMP
    #pragma omp single nowait
    #endif
    {
      
        dst[0][0] = src[0][0];

        for (int j = 1; j < W - 1; j++)
        {
            dst[0][j] = b1 * (src[0][j - 1] + src[0][j + 1]) + b0 * src[0][j];
        }

        dst[0][W - 1] = src[0][W - 1];
    }

     //do first column on CPU, OpenMP if supported

    #ifdef _OPENMP
       #pragma omp for nowait
    #endif

    for (int i = 1; i < H - 1; i++) {
        dst[i][0] *= b1 * (src[i - 1][0] + src[i + 1][0]) + b0 * src[i][0];
                                    }

    //turn array of arrays (the pointer to pointer(s)) into 2D array, excluding the outer rows and columns for newsrc,but  keeping them for oldsrc. Also creating dst for writing.
    float* newsrc = (float*) malloc( (W - 2) * (H - 2) * sizeof(float) ); //allocate the memory for the new src
    float* oldsrc = (float*) malloc( W * H * sizeof(float) ); //allocate the memory for the old src
    float* olddst = (float*) malloc( W * H * sizeof(float) ); //allocate the memory for the new src
    float* newdst = (float*) malloc( W * H * sizeof(float) ); //allocate the memory for the new dst
    int* X = (int*)malloc( (W - 2) * (H - 2) * sizeof(int));
    int* Y = (int*)malloc( (W - 2) * (H - 2) * sizeof(int));

    for (int i = 0; i < H; i++)
	{
	  for (int j = 0; j < W; j++)
	    {
	      oldsrc[i*W + j] = src[i][j];
	      olddst[i*W + j] = dst[i][j];
	    }
	}
    
    for (int i = 1; i < (H - 1); i++)
	{
	  int newi = i - 1;
	  for (int j = 1; j < (W - 1); j++)
	    {
	      int newW = W - 2;
	      int newj = j - 1;
	      newsrc[newi*newW + newj] = src[i][j]; //copy src to newsrc

	      newdst[newi*newW + newj] = dst[i][j]; //copy dst to newdst
	      X[newi*newW + newj] = j;
	      Y[newi*newW + newj] = i;
	    }
	}

    fprintf(stderr, "src pixel 1,1 is %f\n", src[1][1]);
    fprintf(stderr, "newsrc pixel 1,1 is %f\n", newsrc[0]);
    fprintf(stderr, "Checkpoint 3x reached");
    fflush(stderr);

    cl_kernel mykernel;
    kernel_tag gauss_std_tag = gauss3x3std;

    fprintf(stderr, "Checkpoint Charlie reached\n"); fflush(stderr);
    mykernel = helper->reuse_or_create_kernel(gauss_std_tag, "gauss_3x3_inner.cl", "gauss_3x3_inner");
    
    fprintf(stderr, "Checkpoint Maha reached\n"); fflush(stderr);
    
   
     cl_int error_code = NULL;

     //create src and dst on GPU
     cl_mem src_mem_obj = clCreateBuffer(helper->context, CL_MEM_READ_ONLY, (W-2)*(H-2)*sizeof(float), NULL, &error_code);
     fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
        cl_mem oldsrc_mem_obj = clCreateBuffer(helper->context, CL_MEM_READ_ONLY, W*H*sizeof(float), NULL, &error_code);
	fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
	cl_mem olddst_mem_obj = clCreateBuffer(helper->context, CL_MEM_WRITE_ONLY, W*H*sizeof(float), NULL, &error_code);
	fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
     cl_mem ret_mem_obj = clCreateBuffer(helper->context, CL_MEM_READ_WRITE, (W-2)*(H-2)*sizeof(float), NULL, &error_code);
     fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
     /* QUICK HACK  helper->blur_ = ret_mem_obj; */
      cl_mem X_mem_obj = clCreateBuffer(helper->context, CL_MEM_READ_ONLY,(W-2)*(H-2)*sizeof(int), NULL, &error_code);
      fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
         cl_mem Y_mem_obj = clCreateBuffer(helper->context, CL_MEM_READ_ONLY, (W-2)*(H-2)*sizeof(int), NULL, &error_code);
	 fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
     error_code = clEnqueueWriteBuffer(helper->command_queue, src_mem_obj, CL_TRUE, 0, (W-2)*(H-2)*sizeof(float), newsrc, 0, NULL, NULL);
     fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
          error_code = clEnqueueWriteBuffer(helper->command_queue, oldsrc_mem_obj, CL_TRUE, 0, W*H*sizeof(float), oldsrc, 0, NULL, NULL);
	  fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
	   error_code = clEnqueueWriteBuffer(helper->command_queue, olddst_mem_obj, CL_TRUE, 0, W*H*sizeof(float), olddst, 0, NULL, NULL);
	   fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
     error_code = clEnqueueWriteBuffer(helper->command_queue, ret_mem_obj, CL_TRUE, 0, (W-2)*(H-2)*sizeof(float), newdst, 0, NULL, NULL);
     fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
       error_code = clEnqueueWriteBuffer(helper->command_queue, X_mem_obj, CL_TRUE, 0, (W-2)*(H-2)*sizeof(int), X, 0, NULL, NULL);
       fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
          error_code = clEnqueueWriteBuffer(helper->command_queue, Y_mem_obj, CL_TRUE, 0, (W-2)*(H-2)*sizeof(int), Y, 0, NULL, NULL);
	  fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);

     double c0double = c0; double c1double = c1; double c2double = c2;

     error_code = clSetKernelArg(mykernel, 0, sizeof(cl_mem), (void *)&ret_mem_obj);
     fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
     error_code = clSetKernelArg(mykernel, 1, sizeof(cl_mem), (void *)&oldsrc_mem_obj);
     fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
     error_code = clSetKernelArg(mykernel, 2, sizeof(cl_mem), (void *)&X_mem_obj);
     fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
     error_code = clSetKernelArg(mykernel, 3, sizeof(cl_mem), (void *)&Y_mem_obj);
     fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
     error_code = clSetKernelArg(mykernel, 4, sizeof(cl_int), (void *)&W);
     fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
     error_code = clSetKernelArg(mykernel, 5, sizeof(cl_int), (void *)&H);
     fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
     error_code = clSetKernelArg(mykernel, 6, sizeof(cl_double), (void *)&c0double);
     fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
     error_code = clSetKernelArg(mykernel, 7, sizeof(cl_double), (void *)&c1double);
     fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
     error_code = clSetKernelArg(mykernel, 8, sizeof(cl_double), (void *)&c2double);
     fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
     error_code = clSetKernelArg(mykernel, 9, sizeof(cl_mem), (void *)&olddst_mem_obj);
     fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
     
    fprintf(stderr, "Checkpoint 4x reached\n");
    fflush(stderr);
      
     size_t global_item_size = (H-2)*(W-2); 
     size_t local_item_size = 1;
    clock_t diff, diff2;
    clock_t start = clock();
    float *intermediate = (float*)malloc((W-2)*(H-2)*sizeof(float));
     error_code = clEnqueueNDRangeKernel(helper->command_queue, mykernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
      fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
      error_code = clEnqueueReadBuffer(helper->command_queue, ret_mem_obj, CL_TRUE, 0, (W-2)*(H-2)*sizeof(float), intermediate, 0, NULL, NULL);
      fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
      fprintf(stderr, "3x3 read is  %d", error_code); fflush(stderr);

     diff = clock() - start;
     double msec = (double)diff * 1000.0 / (double)CLOCKS_PER_SEC;
     fprintf(stderr, "OpenCL took %f to calculate", msec);
     fflush(stderr);


     free(X); free(Y);

     cl_uint native_double_width;    
clGetDeviceInfo(helper->device_id, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &native_double_width, NULL);

if(native_double_width == 0){
    printf("No double precision support.\n");
}
 else printf("There is double precision support.\n");
 

     // for comparison on CPU
     start = clock();
      for (int i = 1; i < H - 1; i++) {
        dst[i][0] = b1 * (src[i - 1][0] + src[i + 1][0]) + b0 * src[i][0];

        for (int j = 1; j < W - 1; j++) {
            dst[i][j] = c2 * (src[i - 1][j - 1] + src[i - 1][j + 1] + src[i + 1][j - 1] + src[i + 1][j + 1]) + c1 * (src[i - 1][j] + src[i][j - 1] + src[i][j + 1] + src[i + 1][j]) + c0 * src[i][j];
        }

        dst[i][W - 1] = b1 * (src[i - 1][W - 1] + src[i + 1][W - 1]) + b0 * src[i][W - 1];
    }
      diff2 = clock() - start;
      double msec2 = (double)diff2 * 1000.0 / (double)CLOCKS_PER_SEC;
      fprintf(stderr, "CPU took %f to calculate", msec2);
      fprintf(stderr, "\n c2 is %f", c2);
      fprintf(stderr, "\nCPU work says %d,%d is %f; \n", 0, 0, src[0][0]);
      fflush(stderr);
      for (int i = H - 2 - 1; i > H - 2 - 1 - 4; i--) {
	fprintf(stderr, "\n\n\n\n");
	for (int j = 1; j < 4; j++) {
	  fprintf(stderr, "Height is %d\n", H);
	  fprintf(stderr, "Width is %d\n", W);
	  fprintf(stderr, "CPU work says %d,%d is %f; ", i, j, dst[i][j]);
	  fprintf(stderr, "gpu work says %d,%d is %f\n", i, j, intermediate[(i-1)*(W-2) + (j - 1)]);
	  fflush(stderr);
	}
      }
      fprintf(stderr, "CPU word 2nd (%d, 2): %f", (H-1), dst[H-1][2]);

     //copy the memory back to the inner bit of the destination 2D array
    for (int i = 1; i < (H - 2); i++)
	{
	  int newi = i - 1;
	  for (int j = 1; j < (W - 2); j++)
	    {
	      int newW = W - 2;
	      int newj = j - 1;
	      dst[i][j] = intermediate[newi*newW + newj];
	    }
	  
          dst[i][W - 1] = b1 * (src[i - 1][W - 1] + src[i + 1][W - 1]) + b0 * src[i][W - 1];
	}
    fprintf(stderr, "Checkpoint ab reached  \n");
    fflush(stderr);

    fprintf(stderr, "GPU word 2nd (%d, 2): %f", (H-1), dst[H-1][2]);
    
    free(newsrc); free(newdst); free(oldsrc);  free(intermediate);

    

         // last row
#ifdef _OPENMP
    #pragma omp single
#endif
    {
        dst[H - 1][0] = src[H - 1][0];

        for (int j = 1; j < W - 1; j++) {
            dst[H - 1][j] = b1 * (src[H - 1][j - 1] + src[H - 1][j + 1]) + b0 * src[H - 1][j];
	    /*fprintf(stderr, "j: %d \n", j);
	      fflush(stderr);*/
        }
	// fprintf(stderr, "Checkpoint ab2 reached \n");
	//	fflush(stderr);

        dst[H - 1][W - 1] = src[H - 1][W - 1];
    }

    fprintf(stderr, "Checkpoint ab3 reached  \n");
    fflush(stderr);
    
 }

// classical filtering if the support window is small and src != dst
  template<class T> void gauss3x3 (T** RESTRICT src, T** RESTRICT dst, const int W, const int H, const double c0, const double c1, const double c2, const double b0, const double b1) //const T c0, const T c1, const T c2, const T b0, const T b1)
{

    // first row
#ifdef _OPENMP
    #pragma omp single nowait
#endif
    {
      
        dst[0][0] = src[0][0];

        for (int j = 1; j < W - 1; j++)
        {
            dst[0][j] = b1 * (src[0][j - 1] + src[0][j + 1]) + b0 * src[0][j];
        }

        dst[0][W - 1] = src[0][W - 1];
    }

#ifdef _OPENMP
    #pragma omp for nowait
#endif

    for (int i = 1; i < H - 1; i++) {
        dst[i][0] = b1 * (src[i - 1][0] + src[i + 1][0]) + b0 * src[i][0];

        for (int j = 1; j < W - 1; j++) {
            dst[i][j] = c2 * (src[i - 1][j - 1] + src[i - 1][j + 1] + src[i + 1][j - 1] + src[i + 1][j + 1]) + c1 * (src[i - 1][j] + src[i][j - 1] + src[i][j + 1] + src[i + 1][j]) + c0 * src[i][j];
        }

        dst[i][W - 1] = b1 * (src[i - 1][W - 1] + src[i + 1][W - 1]) + b0 * src[i][W - 1];
    }

    // last row
#ifdef _OPENMP
    #pragma omp single
#endif
    {
        dst[H - 1][0] = src[H - 1][0];

        for (int j = 1; j < W - 1; j++) {
            dst[H - 1][j] = b1 * (src[H - 1][j - 1] + src[H - 1][j + 1]) + b0 * src[H - 1][j];
        }

        dst[H - 1][W - 1] = src[H - 1][W - 1];
    }

    
}

// classical filtering if the support window is small and src != dst
template<class T> void gauss3x3mult (T** RESTRICT src, T** RESTRICT dst, const int W, const int H, const double c0, const double c1, const double c2, const double b0, const double b1) //const T c0, const T c1, const T c2, const T b0, const T b1)
{

    // first row
#ifdef _OPENMP
    #pragma omp single nowait
#endif
    {
        dst[0][0] *= src[0][0];

        for (int j = 1; j < W - 1; j++)
        {
            dst[0][j] *= b1 * (src[0][j - 1] + src[0][j + 1]) + b0 * src[0][j];
        }

        dst[0][W - 1] *= src[0][W - 1];
    }

#ifdef _OPENMP
    #pragma omp for nowait
#endif

    for (int i = 1; i < H - 1; i++) {
        dst[i][0] *= b1 * (src[i - 1][0] + src[i + 1][0]) + b0 * src[i][0];

        for (int j = 1; j < W - 1; j++) {
	  dst[i][j] *= c2 * (src[i - 1][j - 1] + src[i - 1][j + 1] + src[i + 1][j - 1] + src[i + 1][j + 1]) + c1 * (src[i - 1][j] + src[i][j - 1] + src[i][j + 1] + src[i + 1][j]) + c0 * src[i][j];
        }

        dst[i][W - 1] *= b1 * (src[i - 1][W - 1] + src[i + 1][W - 1]) + b0 * src[i][W - 1];
    }

    // last row
#ifdef _OPENMP
    #pragma omp single
#endif
    {
        dst[H - 1][0] *= src[H - 1][0];

        for (int j = 1; j < W - 1; j++) {
            dst[H - 1][j] *= b1 * (src[H - 1][j - 1] + src[H - 1][j + 1]) + b0 * src[H - 1][j];
        }

        dst[H - 1][W - 1] *= src[H - 1][W - 1];
    }
}

template<class T> void gauss3x3div (T** RESTRICT src, T** RESTRICT dst, T** RESTRICT divBuffer, const int W, const int H, const double c0, const double c1, const double c2, const double b0, const double b1) { //const T c0, const T c1, const T c2, const T b0, const T b1)

    // first row
#ifdef _OPENMP
    #pragma omp single nowait
#endif
    {
        dst[0][0] = rtengine::max(divBuffer[0][0] / (src[0][0] > 0.f ? src[0][0] : 1.f), 0.f);

        for (int j = 1; j < W - 1; j++)
        {
            float tmp = (b1 * (src[0][j - 1] + src[0][j + 1]) + b0 * src[0][j]);
            dst[0][j] = rtengine::max(divBuffer[0][j] / (tmp > 0.f ? tmp : 1.f), 0.f);
        }

        dst[0][W - 1] = rtengine::max(divBuffer[0][W - 1] / (src[0][W - 1] > 0.f ? src[0][W - 1] : 1.f), 0.f);
    }

#ifdef _OPENMP
    #pragma omp for nowait
#endif

    for (int i = 1; i < H - 1; i++) {
        float tmp = (b1 * (src[i - 1][0] + src[i + 1][0]) + b0 * src[i][0]);
        dst[i][0] = rtengine::max(divBuffer[i][0] / (tmp > 0.f ? tmp : 1.f), 0.f);

        for (int j = 1; j < W - 1; j++) {
            tmp = (c2 * (src[i - 1][j - 1] + src[i - 1][j + 1] + src[i + 1][j - 1] + src[i + 1][j + 1]) + c1 * (src[i - 1][j] + src[i][j - 1] + src[i][j + 1] + src[i + 1][j]) + c0 * src[i][j]);
            dst[i][j] = rtengine::max(divBuffer[i][j] / (tmp > 0.f ? tmp : 1.f), 0.f);
        }

        tmp = (b1 * (src[i - 1][W - 1] + src[i + 1][W - 1]) + b0 * src[i][W - 1]);
        dst[i][W - 1] = rtengine::max(divBuffer[i][W - 1] / (tmp > 0.f ? tmp : 1.f), 0.f);
    }

    // last row
#ifdef _OPENMP
    #pragma omp single
#endif
    {
        dst[H - 1][0] = rtengine::max(divBuffer[H - 1][0] / (src[H - 1][0] > 0.f ? src[H - 1][0] : 1.f), 0.f);

        for (int j = 1; j < W - 1; j++) {
            float tmp = (b1 * (src[H - 1][j - 1] + src[H - 1][j + 1]) + b0 * src[H - 1][j]);
            dst[H - 1][j] = rtengine::max(divBuffer[H - 1][j] / (tmp > 0.f ? tmp : 1.f), 0.f);
        }

        dst[H - 1][W - 1] = rtengine::max(divBuffer[H - 1][W - 1] / (src[H - 1][W - 1] > 0.f ? src[H - 1][W - 1] : 1.f), 0.f);
    }
}

template <class T> void dcdamping (T** RESTRICT aI, T** RESTRICT aO, float damping, int W, int H) {
  
   const float dampingFac = -2.0 / (damping * damping);
   

}

template<class T> void mult_firstrow_and_firstcolumn(T* srcmapped, T* dstmapped,
						    const int W, const int H, const double b0, const double b1)
{
  /** Replicates the first row and first column parts of the gauss3x3mult algorithm above **/
  //First row:
  dstmapped[0]  *=  srcmapped[0];

        for (int j = 1; j < W - 1; j++)
        { 
            dstmapped[j] *= (b1 * (srcmapped[j - 1] + srcmapped[j + 1]) + b0 * srcmapped[j]);
        }

  dstmapped[W - 1] *= srcmapped[W - 1];


  //First column, minus the first and last pixels
        for (int i = 1; i < H - 1; i++)
	{
    dstmapped[W*i] *= (b1 * (srcmapped[W*(i - 1)] + srcmapped[W*(i + 1)]) + b0 * srcmapped[W*i]);;
        }


}

template<class T> void mult_lastcolumn_and_lastrow(T* srcmapped, T* dstmapped,
						    const int W, const int H, const double b0, const double b1)
{
    /** Replicates the last column row and last row parts of the gauss3x3mult standard algorithm above **/
  
  //Last column, minus the first and last pixels
        for (int i = 1; i < H - 1; i++)
	{
          dstmapped[W*i + (W - 1)] *= (b1 * (srcmapped[W*(i - 1) + (W - 1)] + srcmapped[W*(i + 1) + (W - 1)]) + b0 * srcmapped[W*i + (W - 1)]);
        }
  //Last row:
	
	dstmapped[(H-1)*W] *= srcmapped[(H-1)*W] ;

        for (int j = 1; j < W - 1; j++)
        {
	  dstmapped[(H-1)*W + j] *= (b1 * (srcmapped[(H-1)*W + (j - 1)] + srcmapped[(H-1)*W + (j + 1)]) + b0 * srcmapped[(H-1)*W + j]);;
        }

	dstmapped[(H-1)*W + (W-1)] *= srcmapped[(H-1)*W + (W-1)];


}

template<class T> void standard_firstrow_and_firstcolumn(T* srcmapped, T* dstmapped,
						    const int W, const int H, const double b0, const double b1)
{
  /** Replicates the first row and first column parts of the gauss3x3 algorithm above **/
  //First row:
  dstmapped[0]  =  srcmapped[0];

        for (int j = 1; j < W - 1; j++)
        { 
            dstmapped[j] = (b1 * (srcmapped[j - 1] + srcmapped[j + 1]) + b0 * srcmapped[j]);
        }

  dstmapped[W - 1] = srcmapped[W - 1];


  //First column, minus the first and last pixels
        for (int i = 1; i < H - 1; i++)
	{
    dstmapped[W*i] = (b1 * (srcmapped[W*(i - 1)] + srcmapped[W*(i + 1)]) + b0 * srcmapped[W*i]);;
        }


}

template<class T> void standard_lastcolumn_and_lastrow(T* srcmapped, T* dstmapped,
						    const int W, const int H, const double b0, const double b1)
{
    /** Replicates the last column row and last row parts of the gauss3x3 standard algorithm above **/
  
  //Last column, minus the first and last pixels
        for (int i = 1; i < H - 1; i++)
	{
          dstmapped[W*i + (W - 1)] = (b1 * (srcmapped[W*(i - 1) + (W - 1)] + srcmapped[W*(i + 1) + (W - 1)]) + b0 * srcmapped[W*i + (W - 1)]);
        }
  //Last row:
	
	dstmapped[(H-1)*W]  = srcmapped[(H-1)*W] ;

        for (int j = 1; j < W - 1; j++)
        {
	  dstmapped[(H-1)*W + j] = (b1 * (srcmapped[(H-1)*W + (j - 1)] + srcmapped[(H-1)*W + (j + 1)]) + b0 * srcmapped[(H-1)*W + j]);;
        }

	dstmapped[(H-1)*W + (W-1)] = srcmapped[(H-1)*W + (W-1)];


}

template<class T> void div_firstrow_and_firstcolumn(T* srcmapped, T* dstmapped, T** RESTRICT divbuff,
						    const int W, const int H, const double b0, const double b1)
{
  /** Replicates the first row and first column parts of the gauss3x3div algorithm above **/
  //First row:
   dstmapped[0]  = rtengine::max(divbuff[0][0] / (srcmapped[0] > 0.f ? srcmapped[0] : 1.f), 0.f);

        for (int j = 1; j < W - 1; j++)
        {
            float tmp = (b1 * (srcmapped[j - 1] + srcmapped[j + 1]) + b0 * srcmapped[j]);
            dstmapped[j] = rtengine::max(divbuff[0][j] / (tmp > 0.f ? tmp : 1.f), 0.f);
        }

    dstmapped[W - 1] = rtengine::max(divbuff[0][W - 1] / (srcmapped[W - 1] > 0.f ? srcmapped[W - 1] : 1.f), 0.f);


  //First column, minus the first and last pixels
        for (int i = 1; i < H - 1; i++)
	{
	  float tmp = (b1 * (srcmapped[W*(i - 1)] + srcmapped[W*(i + 1)]) + b0 * srcmapped[W*i]);
    dstmapped[W*i] = rtengine::max(divbuff[i][0] / (tmp > 0.f ? tmp : 1.f), 0.f);
        }


}

template<class T> void div_lastcolumn_and_lastrow(T* srcmapped, T* dstmapped, T** RESTRICT divbuff,
						    const int W, const int H, const double b0, const double b1)
{
    /** Replicates the last column row and last row parts of the gauss3x3div algorithm above **/
  
  //Last column, minus the first and last pixels
        for (int i = 1; i < H - 1; i++)
	{
	  float tmp = (b1 * (srcmapped[W*(i - 1) + (W - 1)] + srcmapped[W*(i + 1) + (W - 1)]) + b0 * srcmapped[W*i + (W - 1)]);
          dstmapped[W*i + (W - 1)] = rtengine::max(divbuff[i][W - 1] / (tmp > 0.f ? tmp : 1.f), 0.f);
        }
  //Last row:
	
	dstmapped[(H-1)*W]  = rtengine::max(divbuff[H - 1][0] / (srcmapped[(H-1)*W] > 0.f ? srcmapped[(H-1)*W] : 1.f), 0.f);

        for (int j = 1; j < W - 1; j++)
        {
	  float tmp = (b1 * (srcmapped[(H-1)*W + (j - 1)] + srcmapped[(H-1)*W + (j + 1)]) + b0 * srcmapped[(H-1)*W + j]);
	  dstmapped[(H-1)*W + j] = rtengine::max(divbuff[H - 1][j] / (tmp > 0.f ? tmp : 1.f), 0.f);
        }

	dstmapped[(H-1)*W + (W-1)] = rtengine::max(divbuff[H - 1][W - 1] / (srcmapped[(H-1)*W + (W-1)] > 0.f ? srcmapped[(H-1)*W + (W-1)] : 1.f), 0.f);




}

// use separated filter if the support window is small and src == dst
template<class T> void gaussHorizontal3 (T** src, T** dst, int W, int H, const float c0, const float c1)
{
    T temp[W] ALIGNED16;
#ifdef _OPENMP
    #pragma omp for
#endif

    for (int i = 0; i < H; i++) {
        for (int j = 1; j < W - 1; j++) {
            temp[j] = (T)(c1 * (src[i][j - 1] + src[i][j + 1]) + c0 * src[i][j]);
        }

        dst[i][0] = src[i][0];
        memcpy (dst[i] + 1, temp + 1, (W - 2)*sizeof(T));

        dst[i][W - 1] = src[i][W - 1];
    }
}

#ifdef __SSE2__
template<class T> void gaussVertical3 (T** src, T** dst, int W, int H, const float c0, const float c1)
{
    vfloat Tv = F2V(0.f), Tm1v, Tp1v;
    vfloat Tv1 = F2V(0.f), Tm1v1, Tp1v1;
    vfloat c0v, c1v;
    c0v = F2V(c0);
    c1v = F2V(c1);

#ifdef _OPENMP
    #pragma omp for nowait
#endif

    // process 8 columns per iteration for better usage of cpu cache
    for (int i = 0; i < W - 7; i += 8) {
        Tm1v = LVFU( src[0][i] );
        Tm1v1 = LVFU( src[0][i + 4] );
        STVFU( dst[0][i], Tm1v);
        STVFU( dst[0][i + 4], Tm1v1);

        if (H > 1) {
            Tv = LVFU( src[1][i]);
            Tv1 = LVFU( src[1][i + 4]);
        }

        for (int j = 1; j < H - 1; j++) {
            Tp1v = LVFU( src[j + 1][i]);
            Tp1v1 = LVFU( src[j + 1][i + 4]);
            STVFU( dst[j][i], c1v * (Tp1v + Tm1v) + Tv * c0v);
            STVFU( dst[j][i + 4], c1v * (Tp1v1 + Tm1v1) + Tv1 * c0v);
            Tm1v = Tv;
            Tm1v1 = Tv1;
            Tv = Tp1v;
            Tv1 = Tp1v1;
        }

        STVFU( dst[H - 1][i], LVFU( src[H - 1][i]));
        STVFU( dst[H - 1][i + 4], LVFU( src[H - 1][i + 4]));
    }

// Borders are done without SSE
    float temp[H] ALIGNED16;
#ifdef _OPENMP
    #pragma omp single
#endif

    for (int i = W - (W % 8); i < W; i++) {
        for (int j = 1; j < H - 1; j++) {
            temp[j] = c1 * (src[j - 1][i] + src[j + 1][i]) + c0 * src[j][i];
        }

        dst[0][i] = src[0][i];

        for (int j = 1; j < H - 1; j++) {
            dst[j][i] = temp[j];
        }

        dst[H - 1][i] = src[H - 1][i];
    }
}
#else
template<class T> void gaussVertical3 (T** src, T** dst, int W, int H, const float c0, const float c1)
{
    T temp[H] ALIGNED16;
#ifdef _OPENMP
    #pragma omp for
#endif

    for (int i = 0; i < W; i++) {
        for (int j = 1; j < H - 1; j++) {
            temp[j] = (T)(c1 * (src[j - 1][i] + src[j + 1][i]) + c0 * src[j][i]);
        }

        dst[0][i] = src[0][i];

        for (int j = 1; j < H - 1; j++) {
            dst[j][i] = temp[j];
        }

        dst[H - 1][i] = src[H - 1][i];
    }
}
#endif

#ifdef __SSE2__
// fast gaussian approximation if the support window is large
template<class T> void gaussHorizontalSse (T** src, T** dst, const int W, const int H, const float sigma)
{
    double b1, b2, b3, B, M[3][3];
    calculateYvVFactors<double>(sigma, b1, b2, b3, B, M);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            M[i][j] *= (1.0 + b2 + (b1 - b3) * b3);
            M[i][j] /= (1.0 + b1 - b2 + b3) * (1.0 - b1 - b2 - b3);
        }

    vfloat Rv;
    vfloat Tv, Tm2v, Tm3v;
    vfloat Bv, b1v, b2v, b3v;
    vfloat temp2W, temp2Wp1;
    float tmp[W][4] ALIGNED16;
    Bv = F2V(B);
    b1v = F2V(b1);
    b2v = F2V(b2);
    b3v = F2V(b3);

#ifdef _OPENMP
    #pragma omp for nowait
#endif

    for (int i = 0; i < H - 3; i += 4) {
        Tv = _mm_set_ps(src[i][0], src[i + 1][0], src[i + 2][0], src[i + 3][0]);
        Tm3v = Tv * (Bv + b1v + b2v + b3v);
        STVF( tmp[0][0], Tm3v );

        Tm2v = _mm_set_ps(src[i][1], src[i + 1][1], src[i + 2][1], src[i + 3][1]) * Bv + Tm3v * b1v + Tv * (b2v + b3v);
        STVF( tmp[1][0], Tm2v );

        Rv = _mm_set_ps(src[i][2], src[i + 1][2], src[i + 2][2], src[i + 3][2]) * Bv + Tm2v * b1v + Tm3v * b2v + Tv * b3v;
        STVF( tmp[2][0], Rv );

        for (int j = 3; j < W; j++) {
            Tv = Rv;
            Rv = _mm_set_ps(src[i][j], src[i + 1][j], src[i + 2][j], src[i + 3][j]) * Bv + Tv * b1v + Tm2v * b2v + Tm3v * b3v;
            STVF( tmp[j][0], Rv );
            Tm3v = Tm2v;
            Tm2v = Tv;
        }

        Tv = _mm_set_ps(src[i][W - 1], src[i + 1][W - 1], src[i + 2][W - 1], src[i + 3][W - 1]);

        temp2Wp1 = Tv + F2V(M[2][0]) * (Rv - Tv) + F2V(M[2][1]) * ( Tm2v - Tv ) +  F2V(M[2][2]) * (Tm3v - Tv);
        temp2W = Tv + F2V(M[1][0]) * (Rv - Tv) + F2V(M[1][1]) * (Tm2v - Tv) + F2V(M[1][2]) * (Tm3v - Tv);

        Rv = Tv + F2V(M[0][0]) * (Rv - Tv) + F2V(M[0][1]) * (Tm2v - Tv) + F2V(M[0][2]) * (Tm3v - Tv);
        STVF(tmp[W - 1][0], Rv);

        Tm2v = Bv * Tm2v + b1v * Rv + b2v * temp2W + b3v * temp2Wp1;
        STVF(tmp[W - 2][0], Tm2v);

        Tm3v = Bv * Tm3v + b1v * Tm2v + b2v * Rv + b3v * temp2W;
        STVF(tmp[W - 3][0], Tm3v);

        Tv = Rv;
        Rv = Tm3v;
        Tm3v = Tv;

        for (int j = W - 4; j >= 0; j--) {
            Tv = Rv;
            Rv = LVF(tmp[j][0]) * Bv + Tv * b1v + Tm2v * b2v + Tm3v * b3v;
            STVF(tmp[j][0], Rv);
            Tm3v = Tm2v;
            Tm2v = Tv;
        }

        for (int j = 0; j < W; j++) {
            dst[i + 3][j] = tmp[j][0];
            dst[i + 2][j] = tmp[j][1];
            dst[i + 1][j] = tmp[j][2];
            dst[i + 0][j] = tmp[j][3];
        }


    }

// Borders are done without SSE
#ifdef _OPENMP
    #pragma omp single
#endif

    for (int i = H - (H % 4); i < H; i++) {
        tmp[0][0] = src[i][0] * (B + b1 + b2 + b3);
        tmp[1][0] = B * src[i][1] + b1 * tmp[0][0]  + src[i][0] * (b2 + b3);
        tmp[2][0] = B * src[i][2] + b1 * tmp[1][0]  + b2 * tmp[0][0]  + b3 * src[i][0];

        for (int j = 3; j < W; j++) {
            tmp[j][0] = B * src[i][j] + b1 * tmp[j - 1][0] + b2 * tmp[j - 2][0] + b3 * tmp[j - 3][0];
        }

        float temp2Wm1 = src[i][W - 1] + M[0][0] * (tmp[W - 1][0] - src[i][W - 1]) + M[0][1] * (tmp[W - 2][0] - src[i][W - 1]) + M[0][2] * (tmp[W - 3][0] - src[i][W - 1]);
        float temp2W   = src[i][W - 1] + M[1][0] * (tmp[W - 1][0] - src[i][W - 1]) + M[1][1] * (tmp[W - 2][0] - src[i][W - 1]) + M[1][2] * (tmp[W - 3][0] - src[i][W - 1]);
        float temp2Wp1 = src[i][W - 1] + M[2][0] * (tmp[W - 1][0] - src[i][W - 1]) + M[2][1] * (tmp[W - 2][0] - src[i][W - 1]) + M[2][2] * (tmp[W - 3][0] - src[i][W - 1]);

        tmp[W - 1][0] = temp2Wm1;
        tmp[W - 2][0] = B * tmp[W - 2][0] + b1 * tmp[W - 1][0] + b2 * temp2W + b3 * temp2Wp1;
        tmp[W - 3][0] = B * tmp[W - 3][0] + b1 * tmp[W - 2][0] + b2 * tmp[W - 1][0] + b3 * temp2W;

        for (int j = W - 4; j >= 0; j--) {
            tmp[j][0] = B * tmp[j][0] + b1 * tmp[j + 1][0] + b2 * tmp[j + 2][0] + b3 * tmp[j + 3][0];
        }

        for (int j = 0; j < W; j++) {
            dst[i][j] = tmp[j][0];
        }
    }
}
#endif

// fast gaussian approximation if the support window is large
template<class T> void gaussHorizontal (T** src, T** dst, const int W, const int H, const double sigma)
{
    double b1, b2, b3, B, M[3][3];
    calculateYvVFactors<double>(sigma, b1, b2, b3, B, M);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            M[i][j] /= (1.0 + b1 - b2 + b3) * (1.0 + b2 + (b1 - b3) * b3);
        }

    double temp2[W] ALIGNED16;

#ifdef _OPENMP
    #pragma omp for
#endif

    for (int i = 0; i < H; i++) {

        temp2[0] = B * src[i][0] + b1 * src[i][0] + b2 * src[i][0] + b3 * src[i][0];
        temp2[1] = B * src[i][1] + b1 * temp2[0]  + b2 * src[i][0] + b3 * src[i][0];
        temp2[2] = B * src[i][2] + b1 * temp2[1]  + b2 * temp2[0]  + b3 * src[i][0];

        for (int j = 3; j < W; j++) {
            temp2[j] = B * src[i][j] + b1 * temp2[j - 1] + b2 * temp2[j - 2] + b3 * temp2[j - 3];
        }

        double temp2Wm1 = src[i][W - 1] + M[0][0] * (temp2[W - 1] - src[i][W - 1]) + M[0][1] * (temp2[W - 2] - src[i][W - 1]) + M[0][2] * (temp2[W - 3] - src[i][W - 1]);
        double temp2W   = src[i][W - 1] + M[1][0] * (temp2[W - 1] - src[i][W - 1]) + M[1][1] * (temp2[W - 2] - src[i][W - 1]) + M[1][2] * (temp2[W - 3] - src[i][W - 1]);
        double temp2Wp1 = src[i][W - 1] + M[2][0] * (temp2[W - 1] - src[i][W - 1]) + M[2][1] * (temp2[W - 2] - src[i][W - 1]) + M[2][2] * (temp2[W - 3] - src[i][W - 1]);

        temp2[W - 1] = temp2Wm1;
        temp2[W - 2] = B * temp2[W - 2] + b1 * temp2[W - 1] + b2 * temp2W + b3 * temp2Wp1;
        temp2[W - 3] = B * temp2[W - 3] + b1 * temp2[W - 2] + b2 * temp2[W - 1] + b3 * temp2W;

        for (int j = W - 4; j >= 0; j--) {
            temp2[j] = B * temp2[j] + b1 * temp2[j + 1] + b2 * temp2[j + 2] + b3 * temp2[j + 3];
        }

        for (int j = 0; j < W; j++) {
            dst[i][j] = (T)temp2[j];
        }

    }
}

#ifdef __SSE2__
template<class T> void gaussVerticalSse (T** src, T** dst, const int W, const int H, const float sigma)
{
    double b1, b2, b3, B, M[3][3];
    calculateYvVFactors<double>(sigma, b1, b2, b3, B, M);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            M[i][j] *= (1.0 + b2 + (b1 - b3) * b3);
            M[i][j] /= (1.0 + b1 - b2 + b3) * (1.0 - b1 - b2 - b3);
        }

    float tmp[H][8] ALIGNED16;
    vfloat Rv;
    vfloat Tv, Tm2v, Tm3v;
    vfloat Rv1;
    vfloat Tv1, Tm2v1, Tm3v1;
    vfloat Bv, b1v, b2v, b3v;
    vfloat temp2W, temp2Wp1;
    vfloat temp2W1, temp2Wp11;
    Bv = F2V(B);
    b1v = F2V(b1);
    b2v = F2V(b2);
    b3v = F2V(b3);

#ifdef _OPENMP
    #pragma omp for nowait
#endif

    // process 8 columns per iteration for better usage of cpu cache
    for (int i = 0; i < W - 7; i += 8) {
        Tv = LVFU( src[0][i]);
        Tv1 = LVFU( src[0][i + 4]);
        Rv = Tv * (Bv + b1v + b2v + b3v);
        Rv1 = Tv1 * (Bv + b1v + b2v + b3v);
        Tm3v = Rv;
        Tm3v1 = Rv1;
        STVF( tmp[0][0], Rv );
        STVF( tmp[0][4], Rv1 );

        Rv = LVFU(src[1][i]) * Bv + Rv * b1v + Tv * (b2v + b3v);
        Rv1 = LVFU(src[1][i + 4]) * Bv + Rv1 * b1v + Tv1 * (b2v + b3v);
        Tm2v = Rv;
        Tm2v1 = Rv1;
        STVF( tmp[1][0], Rv );
        STVF( tmp[1][4], Rv1 );

        Rv = LVFU(src[2][i]) * Bv + Rv * b1v + Tm3v * b2v + Tv * b3v;
        Rv1 = LVFU(src[2][i + 4]) * Bv + Rv1 * b1v + Tm3v1 * b2v + Tv1 * b3v;
        STVF( tmp[2][0], Rv );
        STVF( tmp[2][4], Rv1 );

        for (int j = 3; j < H; j++) {
            Tv = Rv;
            Tv1 = Rv1;
            Rv = LVFU(src[j][i]) * Bv +  Tv * b1v + Tm2v * b2v + Tm3v * b3v;
            Rv1 = LVFU(src[j][i + 4]) * Bv +  Tv1 * b1v + Tm2v1 * b2v + Tm3v1 * b3v;
            STVF( tmp[j][0], Rv );
            STVF( tmp[j][4], Rv1 );
            Tm3v = Tm2v;
            Tm3v1 = Tm2v1;
            Tm2v = Tv;
            Tm2v1 = Tv1;
        }

        Tv = LVFU(src[H - 1][i]);
        Tv1 = LVFU(src[H - 1][i + 4]);

        temp2Wp1 = Tv + F2V(M[2][0]) * (Rv - Tv) + F2V(M[2][1]) * (Tm2v - Tv) + F2V(M[2][2]) * (Tm3v - Tv);
        temp2Wp11 = Tv1 + F2V(M[2][0]) * (Rv1 - Tv1) + F2V(M[2][1]) * (Tm2v1 - Tv1) + F2V(M[2][2]) * (Tm3v1 - Tv1);
        temp2W = Tv + F2V(M[1][0]) * (Rv - Tv) + F2V(M[1][1]) * (Tm2v - Tv) + F2V(M[1][2]) * (Tm3v - Tv);
        temp2W1 = Tv1 + F2V(M[1][0]) * (Rv1 - Tv1) + F2V(M[1][1]) * (Tm2v1 - Tv1) + F2V(M[1][2]) * (Tm3v1 - Tv1);

        Rv = Tv + F2V(M[0][0]) * (Rv - Tv) + F2V(M[0][1]) * (Tm2v - Tv) + F2V(M[0][2]) * (Tm3v - Tv);
        Rv1 = Tv1 + F2V(M[0][0]) * (Rv1 - Tv1) + F2V(M[0][1]) * (Tm2v1 - Tv1) + F2V(M[0][2]) * (Tm3v1 - Tv1);
        STVFU( dst[H - 1][i], Rv );
        STVFU( dst[H - 1][i + 4], Rv1 );

        Tm2v = Bv * Tm2v + b1v * Rv + b2v * temp2W + b3v * temp2Wp1;
        Tm2v1 = Bv * Tm2v1 + b1v * Rv1 + b2v * temp2W1 + b3v * temp2Wp11;
        STVFU( dst[H - 2][i], Tm2v );
        STVFU( dst[H - 2][i + 4], Tm2v1 );

        Tm3v = Bv * Tm3v + b1v * Tm2v + b2v * Rv + b3v * temp2W;
        Tm3v1 = Bv * Tm3v1 + b1v * Tm2v1 + b2v * Rv1 + b3v * temp2W1;
        STVFU( dst[H - 3][i], Tm3v );
        STVFU( dst[H - 3][i + 4], Tm3v1 );

        Tv = Rv;
        Tv1 = Rv1;
        Rv = Tm3v;
        Rv1 = Tm3v1;
        Tm3v = Tv;
        Tm3v1 = Tv1;

        for (int j = H - 4; j >= 0; j--) {
            Tv = Rv;
            Tv1 = Rv1;
            Rv = LVF(tmp[j][0]) * Bv +  Tv * b1v + Tm2v * b2v + Tm3v * b3v;
            Rv1 = LVF(tmp[j][4]) * Bv +  Tv1 * b1v + Tm2v1 * b2v + Tm3v1 * b3v;
            STVFU( dst[j][i], Rv );
            STVFU( dst[j][i + 4], Rv1 );
            Tm3v = Tm2v;
            Tm3v1 = Tm2v1;
            Tm2v = Tv;
            Tm2v1 = Tv1;
        }
    }

// Borders are done without SSE
#ifdef _OPENMP
    #pragma omp single
#endif

    for (int i = W - (W % 8); i < W; i++) {
        tmp[0][0] = src[0][i] * (B + b1 + b2 + b3);
        tmp[1][0] = B * src[1][i] + b1 * tmp[0][0] + src[0][i] * (b2 + b3);
        tmp[2][0] = B * src[2][i] + b1 * tmp[1][0] + b2 * tmp[0][0] + b3 * src[0][i];

        for (int j = 3; j < H; j++) {
            tmp[j][0] = B * src[j][i] + b1 * tmp[j - 1][0] + b2 * tmp[j - 2][0] + b3 * tmp[j - 3][0];
        }

        float temp2Hm1 = src[H - 1][i] + M[0][0] * (tmp[H - 1][0] - src[H - 1][i]) + M[0][1] * (tmp[H - 2][0] - src[H - 1][i]) + M[0][2] * (tmp[H - 3][0] - src[H - 1][i]);
        float temp2H   = src[H - 1][i] + M[1][0] * (tmp[H - 1][0] - src[H - 1][i]) + M[1][1] * (tmp[H - 2][0] - src[H - 1][i]) + M[1][2] * (tmp[H - 3][0] - src[H - 1][i]);
        float temp2Hp1 = src[H - 1][i] + M[2][0] * (tmp[H - 1][0] - src[H - 1][i]) + M[2][1] * (tmp[H - 2][0] - src[H - 1][i]) + M[2][2] * (tmp[H - 3][0] - src[H - 1][i]);

        tmp[H - 1][0] = temp2Hm1;
        tmp[H - 2][0] = B * tmp[H - 2][0] + b1 * tmp[H - 1][0] + b2 * temp2H + b3 * temp2Hp1;
        tmp[H - 3][0] = B * tmp[H - 3][0] + b1 * tmp[H - 2][0] + b2 * tmp[H - 1][0] + b3 * temp2H;

        for (int j = H - 4; j >= 0; j--) {
            tmp[j][0] = B * tmp[j][0] + b1 * tmp[j + 1][0] + b2 * tmp[j + 2][0] + b3 * tmp[j + 3][0];
        }

        for (int j = 0; j < H; j++) {
            dst[j][i] = tmp[j][0];
        }

    }
}
#endif

#ifdef __SSE2__
template<class T> void gaussVerticalSsemult (T** RESTRICT src, T** RESTRICT dst, const int W, const int H, const float sigma)
{
    double b1, b2, b3, B, M[3][3];
    calculateYvVFactors<double>(sigma, b1, b2, b3, B, M);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            M[i][j] *= (1.0 + b2 + (b1 - b3) * b3);
            M[i][j] /= (1.0 + b1 - b2 + b3) * (1.0 - b1 - b2 - b3);
        }

    float tmp[H][8] ALIGNED16;
    vfloat Rv;
    vfloat Tv, Tm2v, Tm3v;
    vfloat Rv1;
    vfloat Tv1, Tm2v1, Tm3v1;
    vfloat Bv, b1v, b2v, b3v;
    vfloat temp2W, temp2Wp1;
    vfloat temp2W1, temp2Wp11;
    Bv = F2V(B);
    b1v = F2V(b1);
    b2v = F2V(b2);
    b3v = F2V(b3);

#ifdef _OPENMP
    #pragma omp for nowait
#endif

    // process 8 columns per iteration for better usage of cpu cache
    for (int i = 0; i < W - 7; i += 8) {
        Tv = LVFU( src[0][i]);
        Tv1 = LVFU( src[0][i + 4]);
        Rv = Tv * (Bv + b1v + b2v + b3v);
        Rv1 = Tv1 * (Bv + b1v + b2v + b3v);
        Tm3v = Rv;
        Tm3v1 = Rv1;
        STVF( tmp[0][0], Rv );
        STVF( tmp[0][4], Rv1 );

        Rv = LVFU(src[1][i]) * Bv + Rv * b1v + Tv * (b2v + b3v);
        Rv1 = LVFU(src[1][i + 4]) * Bv + Rv1 * b1v + Tv1 * (b2v + b3v);
        Tm2v = Rv;
        Tm2v1 = Rv1;
        STVF( tmp[1][0], Rv );
        STVF( tmp[1][4], Rv1 );

        Rv = LVFU(src[2][i]) * Bv + Rv * b1v + Tm3v * b2v + Tv * b3v;
        Rv1 = LVFU(src[2][i + 4]) * Bv + Rv1 * b1v + Tm3v1 * b2v + Tv1 * b3v;
        STVF( tmp[2][0], Rv );
        STVF( tmp[2][4], Rv1 );

        for (int j = 3; j < H; j++) {
            Tv = Rv;
            Tv1 = Rv1;
            Rv = LVFU(src[j][i]) * Bv +  Tv * b1v + Tm2v * b2v + Tm3v * b3v;
            Rv1 = LVFU(src[j][i + 4]) * Bv +  Tv1 * b1v + Tm2v1 * b2v + Tm3v1 * b3v;
            STVF( tmp[j][0], Rv );
            STVF( tmp[j][4], Rv1 );
            Tm3v = Tm2v;
            Tm3v1 = Tm2v1;
            Tm2v = Tv;
            Tm2v1 = Tv1;
        }

        Tv = LVFU(src[H - 1][i]);
        Tv1 = LVFU(src[H - 1][i + 4]);

        temp2Wp1 = Tv + F2V(M[2][0]) * (Rv - Tv) + F2V(M[2][1]) * (Tm2v - Tv) + F2V(M[2][2]) * (Tm3v - Tv);
        temp2Wp11 = Tv1 + F2V(M[2][0]) * (Rv1 - Tv1) + F2V(M[2][1]) * (Tm2v1 - Tv1) + F2V(M[2][2]) * (Tm3v1 - Tv1);
        temp2W = Tv + F2V(M[1][0]) * (Rv - Tv) + F2V(M[1][1]) * (Tm2v - Tv) + F2V(M[1][2]) * (Tm3v - Tv);
        temp2W1 = Tv1 + F2V(M[1][0]) * (Rv1 - Tv1) + F2V(M[1][1]) * (Tm2v1 - Tv1) + F2V(M[1][2]) * (Tm3v1 - Tv1);

        Rv = Tv + F2V(M[0][0]) * (Rv - Tv) + F2V(M[0][1]) * (Tm2v - Tv) + F2V(M[0][2]) * (Tm3v - Tv);
        Rv1 = Tv1 + F2V(M[0][0]) * (Rv1 - Tv1) + F2V(M[0][1]) * (Tm2v1 - Tv1) + F2V(M[0][2]) * (Tm3v1 - Tv1);
        STVFU( dst[H - 1][i], LVFU(dst[H - 1][i]) * Rv );
        STVFU( dst[H - 1][i + 4], LVFU(dst[H - 1][i + 4]) * Rv1 );

        Tm2v = Bv * Tm2v + b1v * Rv + b2v * temp2W + b3v * temp2Wp1;
        Tm2v1 = Bv * Tm2v1 + b1v * Rv1 + b2v * temp2W1 + b3v * temp2Wp11;
        STVFU( dst[H - 2][i], LVFU(dst[H - 2][i]) * Tm2v );
        STVFU( dst[H - 2][i + 4], LVFU(dst[H - 2][i + 4]) * Tm2v1 );

        Tm3v = Bv * Tm3v + b1v * Tm2v + b2v * Rv + b3v * temp2W;
        Tm3v1 = Bv * Tm3v1 + b1v * Tm2v1 + b2v * Rv1 + b3v * temp2W1;
        STVFU( dst[H - 3][i], LVFU(dst[H - 3][i]) * Tm3v );
        STVFU( dst[H - 3][i + 4], LVFU(dst[H - 3][i + 4]) * Tm3v1 );

        Tv = Rv;
        Tv1 = Rv1;
        Rv = Tm3v;
        Rv1 = Tm3v1;
        Tm3v = Tv;
        Tm3v1 = Tv1;

        for (int j = H - 4; j >= 0; j--) {
            Tv = Rv;
            Tv1 = Rv1;
            Rv = LVF(tmp[j][0]) * Bv +  Tv * b1v + Tm2v * b2v + Tm3v * b3v;
            Rv1 = LVF(tmp[j][4]) * Bv +  Tv1 * b1v + Tm2v1 * b2v + Tm3v1 * b3v;
            STVFU( dst[j][i], LVFU(dst[j][i]) * Rv );
            STVFU( dst[j][i + 4], LVFU(dst[j][i + 4]) * Rv1 );
            Tm3v = Tm2v;
            Tm3v1 = Tm2v1;
            Tm2v = Tv;
            Tm2v1 = Tv1;
        }
    }

// Borders are done without SSE
#ifdef _OPENMP
    #pragma omp single
#endif

    for (int i = W - (W % 8); i < W; i++) {
        tmp[0][0] = src[0][i] * (B + b1 + b2 + b3);
        tmp[1][0] = B * src[1][i] + b1 * tmp[0][0] + src[0][i] * (b2 + b3);
        tmp[2][0] = B * src[2][i] + b1 * tmp[1][0] + b2 * tmp[0][0] + b3 * src[0][i];

        for (int j = 3; j < H; j++) {
            tmp[j][0] = B * src[j][i] + b1 * tmp[j - 1][0] + b2 * tmp[j - 2][0] + b3 * tmp[j - 3][0];
        }

        float temp2Hm1 = src[H - 1][i] + M[0][0] * (tmp[H - 1][0] - src[H - 1][i]) + M[0][1] * (tmp[H - 2][0] - src[H - 1][i]) + M[0][2] * (tmp[H - 3][0] - src[H - 1][i]);
        float temp2H   = src[H - 1][i] + M[1][0] * (tmp[H - 1][0] - src[H - 1][i]) + M[1][1] * (tmp[H - 2][0] - src[H - 1][i]) + M[1][2] * (tmp[H - 3][0] - src[H - 1][i]);
        float temp2Hp1 = src[H - 1][i] + M[2][0] * (tmp[H - 1][0] - src[H - 1][i]) + M[2][1] * (tmp[H - 2][0] - src[H - 1][i]) + M[2][2] * (tmp[H - 3][0] - src[H - 1][i]);

        tmp[H - 1][0] = temp2Hm1;
        tmp[H - 2][0] = B * tmp[H - 2][0] + b1 * tmp[H - 1][0] + b2 * temp2H + b3 * temp2Hp1;
        tmp[H - 3][0] = B * tmp[H - 3][0] + b1 * tmp[H - 2][0] + b2 * tmp[H - 1][0] + b3 * temp2H;

        for (int j = H - 4; j >= 0; j--) {
            tmp[j][0] = B * tmp[j][0] + b1 * tmp[j + 1][0] + b2 * tmp[j + 2][0] + b3 * tmp[j + 3][0];
        }

        for (int j = 0; j < H; j++) {
            dst[j][i] *= tmp[j][0];
        }

    }
}

template<class T> void gaussVerticalSsediv (T** src, T** dst, T** divBuffer, const int W, const int H, const float sigma)
{
    double b1, b2, b3, B, M[3][3];
    calculateYvVFactors<double>(sigma, b1, b2, b3, B, M);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            M[i][j] *= (1.0 + b2 + (b1 - b3) * b3);
            M[i][j] /= (1.0 + b1 - b2 + b3) * (1.0 - b1 - b2 - b3);
        }

    float tmp[H][8] ALIGNED16;
    vfloat Rv;
    vfloat Tv, Tm2v, Tm3v;
    vfloat Rv1;
    vfloat Tv1, Tm2v1, Tm3v1;
    vfloat Bv, b1v, b2v, b3v;
    vfloat temp2W, temp2Wp1;
    vfloat temp2W1, temp2Wp11;
    vfloat onev = F2V(1.f);
    Bv = F2V(B);
    b1v = F2V(b1);
    b2v = F2V(b2);
    b3v = F2V(b3);

#ifdef _OPENMP
    #pragma omp for nowait
#endif

    // process 8 columns per iteration for better usage of cpu cache
    for (int i = 0; i < W - 7; i += 8) {
        Tv = LVFU( src[0][i]);
        Tv1 = LVFU( src[0][i + 4]);
        Rv = Tv * (Bv + b1v + b2v + b3v);
        Rv1 = Tv1 * (Bv + b1v + b2v + b3v);
        Tm3v = Rv;
        Tm3v1 = Rv1;
        STVF( tmp[0][0], Rv );
        STVF( tmp[0][4], Rv1 );

        Rv = LVFU(src[1][i]) * Bv + Rv * b1v + Tv * (b2v + b3v);
        Rv1 = LVFU(src[1][i + 4]) * Bv + Rv1 * b1v + Tv1 * (b2v + b3v);
        Tm2v = Rv;
        Tm2v1 = Rv1;
        STVF( tmp[1][0], Rv );
        STVF( tmp[1][4], Rv1 );

        Rv = LVFU(src[2][i]) * Bv + Rv * b1v + Tm3v * b2v + Tv * b3v;
        Rv1 = LVFU(src[2][i + 4]) * Bv + Rv1 * b1v + Tm3v1 * b2v + Tv1 * b3v;
        STVF( tmp[2][0], Rv );
        STVF( tmp[2][4], Rv1 );

        for (int j = 3; j < H; j++) {
            Tv = Rv;
            Tv1 = Rv1;
            Rv = LVFU(src[j][i]) * Bv +  Tv * b1v + Tm2v * b2v + Tm3v * b3v;
            Rv1 = LVFU(src[j][i + 4]) * Bv +  Tv1 * b1v + Tm2v1 * b2v + Tm3v1 * b3v;
            STVF( tmp[j][0], Rv );
            STVF( tmp[j][4], Rv1 );
            Tm3v = Tm2v;
            Tm3v1 = Tm2v1;
            Tm2v = Tv;
            Tm2v1 = Tv1;
        }

        Tv = LVFU(src[H - 1][i]);
        Tv1 = LVFU(src[H - 1][i + 4]);

        temp2Wp1 = Tv + F2V(M[2][0]) * (Rv - Tv) + F2V(M[2][1]) * (Tm2v - Tv) + F2V(M[2][2]) * (Tm3v - Tv);
        temp2Wp11 = Tv1 + F2V(M[2][0]) * (Rv1 - Tv1) + F2V(M[2][1]) * (Tm2v1 - Tv1) + F2V(M[2][2]) * (Tm3v1 - Tv1);
        temp2W = Tv + F2V(M[1][0]) * (Rv - Tv) + F2V(M[1][1]) * (Tm2v - Tv) + F2V(M[1][2]) * (Tm3v - Tv);
        temp2W1 = Tv1 + F2V(M[1][0]) * (Rv1 - Tv1) + F2V(M[1][1]) * (Tm2v1 - Tv1) + F2V(M[1][2]) * (Tm3v1 - Tv1);

        Rv = Tv + F2V(M[0][0]) * (Rv - Tv) + F2V(M[0][1]) * (Tm2v - Tv) + F2V(M[0][2]) * (Tm3v - Tv);
        Rv1 = Tv1 + F2V(M[0][0]) * (Rv1 - Tv1) + F2V(M[0][1]) * (Tm2v1 - Tv1) + F2V(M[0][2]) * (Tm3v1 - Tv1);

        STVFU( dst[H - 1][i], LVFU(divBuffer[H - 1][i]) / vself(vmaskf_gt(Rv, ZEROV), Rv, onev));
        STVFU( dst[H - 1][i + 4], LVFU(divBuffer[H - 1][i + 4]) / vself(vmaskf_gt(Rv1, ZEROV), Rv1, onev));

        Tm2v = Bv * Tm2v + b1v * Rv + b2v * temp2W + b3v * temp2Wp1;
        Tm2v1 = Bv * Tm2v1 + b1v * Rv1 + b2v * temp2W1 + b3v * temp2Wp11;
        STVFU( dst[H - 2][i], LVFU(divBuffer[H - 2][i]) / vself(vmaskf_gt(Tm2v, ZEROV), Tm2v, onev));
        STVFU( dst[H - 2][i + 4], LVFU(divBuffer[H - 2][i + 4]) / vself(vmaskf_gt(Tm2v1, ZEROV), Tm2v1, onev));

        Tm3v = Bv * Tm3v + b1v * Tm2v + b2v * Rv + b3v * temp2W;
        Tm3v1 = Bv * Tm3v1 + b1v * Tm2v1 + b2v * Rv1 + b3v * temp2W1;
        STVFU( dst[H - 3][i], LVFU(divBuffer[H - 3][i]) / vself(vmaskf_gt(Tm3v, ZEROV), Tm3v, onev));
        STVFU( dst[H - 3][i + 4], LVFU(divBuffer[H - 3][i + 4]) / vself(vmaskf_gt(Tm3v1, ZEROV), Tm3v1, onev));

        Tv = Rv;
        Tv1 = Rv1;
        Rv = Tm3v;
        Rv1 = Tm3v1;
        Tm3v = Tv;
        Tm3v1 = Tv1;

        for (int j = H - 4; j >= 0; j--) {
            Tv = Rv;
            Tv1 = Rv1;
            Rv = LVF(tmp[j][0]) * Bv +  Tv * b1v + Tm2v * b2v + Tm3v * b3v;
            Rv1 = LVF(tmp[j][4]) * Bv +  Tv1 * b1v + Tm2v1 * b2v + Tm3v1 * b3v;
            STVFU( dst[j][i], vmaxf(LVFU(divBuffer[j][i]) / vself(vmaskf_gt(Rv, ZEROV), Rv, onev), ZEROV));
            STVFU( dst[j][i + 4], vmaxf(LVFU(divBuffer[j][i + 4]) / vself(vmaskf_gt(Rv1, ZEROV), Rv1, onev), ZEROV));
            Tm3v = Tm2v;
            Tm3v1 = Tm2v1;
            Tm2v = Tv;
            Tm2v1 = Tv1;
        }
    }

// Borders are done without SSE
#ifdef _OPENMP
    #pragma omp single
#endif

    for (int i = W - (W % 8); i < W; i++) {
        tmp[0][0] = src[0][i] * (B + b1 + b2 + b3);
        tmp[1][0] = B * src[1][i] + b1 * tmp[0][0] + src[0][i] * (b2 + b3);
        tmp[2][0] = B * src[2][i] + b1 * tmp[1][0] + b2 * tmp[0][0] + b3 * src[0][i];

        for (int j = 3; j < H; j++) {
            tmp[j][0] = B * src[j][i] + b1 * tmp[j - 1][0] + b2 * tmp[j - 2][0] + b3 * tmp[j - 3][0];
        }

        float temp2Hm1 = src[H - 1][i] + M[0][0] * (tmp[H - 1][0] - src[H - 1][i]) + M[0][1] * (tmp[H - 2][0] - src[H - 1][i]) + M[0][2] * (tmp[H - 3][0] - src[H - 1][i]);
        float temp2H   = src[H - 1][i] + M[1][0] * (tmp[H - 1][0] - src[H - 1][i]) + M[1][1] * (tmp[H - 2][0] - src[H - 1][i]) + M[1][2] * (tmp[H - 3][0] - src[H - 1][i]);
        float temp2Hp1 = src[H - 1][i] + M[2][0] * (tmp[H - 1][0] - src[H - 1][i]) + M[2][1] * (tmp[H - 2][0] - src[H - 1][i]) + M[2][2] * (tmp[H - 3][0] - src[H - 1][i]);

        tmp[H - 1][0] = temp2Hm1;
        tmp[H - 2][0] = B * tmp[H - 2][0] + b1 * tmp[H - 1][0] + b2 * temp2H + b3 * temp2Hp1;
        tmp[H - 3][0] = B * tmp[H - 3][0] + b1 * tmp[H - 2][0] + b2 * tmp[H - 1][0] + b3 * temp2H;

        for (int j = H - 4; j >= 0; j--) {
            tmp[j][0] = B * tmp[j][0] + b1 * tmp[j + 1][0] + b2 * tmp[j + 2][0] + b3 * tmp[j + 3][0];
        }

        for (int j = 0; j < H; j++) {
            dst[j][i] = rtengine::max(divBuffer[j][i] / (tmp[j][0] > 0.f ? tmp[j][0] : 1.f), 0.f);
        }

    }
}

#endif

template<class T> void gaussVertical (T** src, T** dst, const int W, const int H, const double sigma)
{
    double b1, b2, b3, B, M[3][3];
    calculateYvVFactors<double>(sigma, b1, b2, b3, B, M);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            M[i][j] /= (1.0 + b1 - b2 + b3) * (1.0 + b2 + (b1 - b3) * b3);
        }

    // process 'numcols' columns for better usage of L1 cpu cache (especially faster for large values of H)
    static const int numcols = 8;
    double temp2[H][numcols] ALIGNED16;
    double temp2Hm1[numcols], temp2H[numcols], temp2Hp1[numcols];
#ifdef _OPENMP
    #pragma omp for nowait
#endif

    for (unsigned int i = 0; i < static_cast<unsigned>(std::max(0, W - numcols + 1)); i += numcols) {
        for (int k = 0; k < numcols; k++) {
            temp2[0][k] = B * src[0][i + k] + b1 * src[0][i + k] + b2 * src[0][i + k] + b3 * src[0][i + k];
            temp2[1][k] = B * src[1][i + k] + b1 * temp2[0][k] + b2 * src[0][i + k] + b3 * src[0][i + k];
            temp2[2][k] = B * src[2][i + k] + b1 * temp2[1][k] + b2 * temp2[0][k] + b3 * src[0][i + k];
        }

        for (int j = 3; j < H; j++) {
            for (int k = 0; k < numcols; k++) {
                temp2[j][k] = B * src[j][i + k] + b1 * temp2[j - 1][k] + b2 * temp2[j - 2][k] + b3 * temp2[j - 3][k];
            }
        }

        for (int k = 0; k < numcols; k++) {
            temp2Hm1[k] = src[H - 1][i + k] + M[0][0] * (temp2[H - 1][k] - src[H - 1][i + k]) + M[0][1] * (temp2[H - 2][k] - src[H - 1][i + k]) + M[0][2] * (temp2[H - 3][k] - src[H - 1][i + k]);
            temp2H[k]   = src[H - 1][i + k] + M[1][0] * (temp2[H - 1][k] - src[H - 1][i + k]) + M[1][1] * (temp2[H - 2][k] - src[H - 1][i + k]) + M[1][2] * (temp2[H - 3][k] - src[H - 1][i + k]);
            temp2Hp1[k] = src[H - 1][i + k] + M[2][0] * (temp2[H - 1][k] - src[H - 1][i + k]) + M[2][1] * (temp2[H - 2][k] - src[H - 1][i + k]) + M[2][2] * (temp2[H - 3][k] - src[H - 1][i + k]);
        }

        for (int k = 0; k < numcols; k++) {
            dst[H - 1][i + k] = temp2[H - 1][k] = temp2Hm1[k];
            dst[H - 2][i + k] = temp2[H - 2][k] = B * temp2[H - 2][k] + b1 * temp2[H - 1][k] + b2 * temp2H[k] + b3 * temp2Hp1[k];
            dst[H - 3][i + k] = temp2[H - 3][k] = B * temp2[H - 3][k] + b1 * temp2[H - 2][k] + b2 * temp2[H - 1][k] + b3 * temp2H[k];
        }

        for (int j = H - 4; j >= 0; j--) {
            for (int k = 0; k < numcols; k++) {
                dst[j][i + k] = temp2[j][k] = B * temp2[j][k] + b1 * temp2[j + 1][k] + b2 * temp2[j + 2][k] + b3 * temp2[j + 3][k];
            }
        }
    }

#ifdef _OPENMP
    #pragma omp single
#endif

    // process remaining columns
    for (int i = W - (W % numcols); i < W; i++) {
        temp2[0][0] = B * src[0][i] + b1 * src[0][i] + b2 * src[0][i] + b3 * src[0][i];
        temp2[1][0] = B * src[1][i] + b1 * temp2[0][0]  + b2 * src[0][i] + b3 * src[0][i];
        temp2[2][0] = B * src[2][i] + b1 * temp2[1][0]  + b2 * temp2[0][0]  + b3 * src[0][i];

        for (int j = 3; j < H; j++) {
            temp2[j][0] = B * src[j][i] + b1 * temp2[j - 1][0] + b2 * temp2[j - 2][0] + b3 * temp2[j - 3][0];
        }

        double temp2Hm1 = src[H - 1][i] + M[0][0] * (temp2[H - 1][0] - src[H - 1][i]) + M[0][1] * (temp2[H - 2][0] - src[H - 1][i]) + M[0][2] * (temp2[H - 3][0] - src[H - 1][i]);
        double temp2H   = src[H - 1][i] + M[1][0] * (temp2[H - 1][0] - src[H - 1][i]) + M[1][1] * (temp2[H - 2][0] - src[H - 1][i]) + M[1][2] * (temp2[H - 3][0] - src[H - 1][i]);
        double temp2Hp1 = src[H - 1][i] + M[2][0] * (temp2[H - 1][0] - src[H - 1][i]) + M[2][1] * (temp2[H - 2][0] - src[H - 1][i]) + M[2][2] * (temp2[H - 3][0] - src[H - 1][i]);

        dst[H - 1][i] = temp2[H - 1][0] = temp2Hm1;
        dst[H - 2][i] = temp2[H - 2][0] = B * temp2[H - 2][0] + b1 * temp2[H - 1][0] + b2 * temp2H + b3 * temp2Hp1;
        dst[H - 3][i] = temp2[H - 3][0] = B * temp2[H - 3][0] + b1 * temp2[H - 2][0] + b2 * temp2[H - 1][0] + b3 * temp2H;

        for (int j = H - 4; j >= 0; j--) {
            dst[j][i] = temp2[j][0] = B * temp2[j][0] + b1 * temp2[j + 1][0] + b2 * temp2[j + 2][0] + b3 * temp2[j + 3][0];
        }
    }
}

#ifndef __SSE2__
template<class T> void gaussVerticaldiv (T** src, T** dst, T** divBuffer, const int W, const int H, const double sigma)
{
    double b1, b2, b3, B, M[3][3];
    calculateYvVFactors<double>(sigma, b1, b2, b3, B, M);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            M[i][j] /= (1.0 + b1 - b2 + b3) * (1.0 + b2 + (b1 - b3) * b3);
        }

    // process 'numcols' columns for better usage of L1 cpu cache (especially faster for large values of H)
    static const int numcols = 8;
    double temp2[H][numcols] ALIGNED16;
    double temp2Hm1[numcols], temp2H[numcols], temp2Hp1[numcols];
#ifdef _OPENMP
    #pragma omp for nowait
#endif

    for (int i = 0; i < W - numcols + 1; i += numcols) {
        for (int k = 0; k < numcols; k++) {
            temp2[0][k] = B * src[0][i + k] + b1 * src[0][i + k] + b2 * src[0][i + k] + b3 * src[0][i + k];
            temp2[1][k] = B * src[1][i + k] + b1 * temp2[0][k] + b2 * src[0][i + k] + b3 * src[0][i + k];
            temp2[2][k] = B * src[2][i + k] + b1 * temp2[1][k] + b2 * temp2[0][k] + b3 * src[0][i + k];
        }

        for (int j = 3; j < H; j++) {
            for (int k = 0; k < numcols; k++) {
                temp2[j][k] = B * src[j][i + k] + b1 * temp2[j - 1][k] + b2 * temp2[j - 2][k] + b3 * temp2[j - 3][k];
            }
        }

        for (int k = 0; k < numcols; k++) {
            temp2Hm1[k] = src[H - 1][i + k] + M[0][0] * (temp2[H - 1][k] - src[H - 1][i + k]) + M[0][1] * (temp2[H - 2][k] - src[H - 1][i + k]) + M[0][2] * (temp2[H - 3][k] - src[H - 1][i + k]);
            temp2H[k]   = src[H - 1][i + k] + M[1][0] * (temp2[H - 1][k] - src[H - 1][i + k]) + M[1][1] * (temp2[H - 2][k] - src[H - 1][i + k]) + M[1][2] * (temp2[H - 3][k] - src[H - 1][i + k]);
            temp2Hp1[k] = src[H - 1][i + k] + M[2][0] * (temp2[H - 1][k] - src[H - 1][i + k]) + M[2][1] * (temp2[H - 2][k] - src[H - 1][i + k]) + M[2][2] * (temp2[H - 3][k] - src[H - 1][i + k]);
        }

        for (int k = 0; k < numcols; k++) {
            dst[H - 1][i + k] = rtengine::max(divBuffer[H - 1][i + k] / (temp2[H - 1][k] = temp2Hm1[k]), 0.0);
            dst[H - 2][i + k] = rtengine::max(divBuffer[H - 2][i + k] / (temp2[H - 2][k] = B * temp2[H - 2][k] + b1 * temp2[H - 1][k] + b2 * temp2H[k] + b3 * temp2Hp1[k]), 0.0);
            dst[H - 3][i + k] = rtengine::max(divBuffer[H - 3][i + k] / (temp2[H - 3][k] = B * temp2[H - 3][k] + b1 * temp2[H - 2][k] + b2 * temp2[H - 1][k] + b3 * temp2H[k]), 0.0);
        }

        for (int j = H - 4; j >= 0; j--) {
            for (int k = 0; k < numcols; k++) {
                dst[j][i + k] = rtengine::max(divBuffer[j][i + k] / (temp2[j][k] = B * temp2[j][k] + b1 * temp2[j + 1][k] + b2 * temp2[j + 2][k] + b3 * temp2[j + 3][k]), 0.0);
            }
        }
    }

#ifdef _OPENMP
    #pragma omp single
#endif

    // process remaining columns
    for (int i = W - (W % numcols); i < W; i++) {
        temp2[0][0] = B * src[0][i] + b1 * src[0][i] + b2 * src[0][i] + b3 * src[0][i];
        temp2[1][0] = B * src[1][i] + b1 * temp2[0][0]  + b2 * src[0][i] + b3 * src[0][i];
        temp2[2][0] = B * src[2][i] + b1 * temp2[1][0]  + b2 * temp2[0][0]  + b3 * src[0][i];

        for (int j = 3; j < H; j++) {
            temp2[j][0] = B * src[j][i] + b1 * temp2[j - 1][0] + b2 * temp2[j - 2][0] + b3 * temp2[j - 3][0];
        }

        double temp2Hm1 = src[H - 1][i] + M[0][0] * (temp2[H - 1][0] - src[H - 1][i]) + M[0][1] * (temp2[H - 2][0] - src[H - 1][i]) + M[0][2] * (temp2[H - 3][0] - src[H - 1][i]);
        double temp2H   = src[H - 1][i] + M[1][0] * (temp2[H - 1][0] - src[H - 1][i]) + M[1][1] * (temp2[H - 2][0] - src[H - 1][i]) + M[1][2] * (temp2[H - 3][0] - src[H - 1][i]);
        double temp2Hp1 = src[H - 1][i] + M[2][0] * (temp2[H - 1][0] - src[H - 1][i]) + M[2][1] * (temp2[H - 2][0] - src[H - 1][i]) + M[2][2] * (temp2[H - 3][0] - src[H - 1][i]);

        dst[H - 1][i] = rtengine::max(divBuffer[H - 1][i] / (temp2[H - 1][0] = temp2Hm1), 0.0);
        dst[H - 2][i] = rtengine::max(divBuffer[H - 2][i] / (temp2[H - 2][0] = B * temp2[H - 2][0] + b1 * temp2[H - 1][0] + b2 * temp2H + b3 * temp2Hp1), 0.0);
        dst[H - 3][i] = rtengine::max(divBuffer[H - 3][i] / (temp2[H - 3][0] = B * temp2[H - 3][0] + b1 * temp2[H - 2][0] + b2 * temp2[H - 1][0] + b3 * temp2H), 0.0);

        for (int j = H - 4; j >= 0; j--) {
            dst[j][i] = rtengine::max(divBuffer[j][i] / (temp2[j][0] = B * temp2[j][0] + b1 * temp2[j + 1][0] + b2 * temp2[j + 2][0] + b3 * temp2[j + 3][0]), 0.0);
        }
    }
}

template<class T> void gaussVerticalmult (T** src, T** dst, const int W, const int H, const double sigma)
{
    double b1, b2, b3, B, M[3][3];
    calculateYvVFactors<double>(sigma, b1, b2, b3, B, M);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            M[i][j] /= (1.0 + b1 - b2 + b3) * (1.0 + b2 + (b1 - b3) * b3);
        }

    // process 'numcols' columns for better usage of L1 cpu cache (especially faster for large values of H)
    static const int numcols = 8;
    double temp2[H][numcols] ALIGNED16;
    double temp2Hm1[numcols], temp2H[numcols], temp2Hp1[numcols];
#ifdef _OPENMP
    #pragma omp for nowait
#endif

    for (int i = 0; i < W - numcols + 1; i += numcols) {
        for (int k = 0; k < numcols; k++) {
            temp2[0][k] = B * src[0][i + k] + b1 * src[0][i + k] + b2 * src[0][i + k] + b3 * src[0][i + k];
            temp2[1][k] = B * src[1][i + k] + b1 * temp2[0][k] + b2 * src[0][i + k] + b3 * src[0][i + k];
            temp2[2][k] = B * src[2][i + k] + b1 * temp2[1][k] + b2 * temp2[0][k] + b3 * src[0][i + k];
        }

        for (int j = 3; j < H; j++) {
            for (int k = 0; k < numcols; k++) {
                temp2[j][k] = B * src[j][i + k] + b1 * temp2[j - 1][k] + b2 * temp2[j - 2][k] + b3 * temp2[j - 3][k];
            }
        }

        for (int k = 0; k < numcols; k++) {
            temp2Hm1[k] = src[H - 1][i + k] + M[0][0] * (temp2[H - 1][k] - src[H - 1][i + k]) + M[0][1] * (temp2[H - 2][k] - src[H - 1][i + k]) + M[0][2] * (temp2[H - 3][k] - src[H - 1][i + k]);
            temp2H[k]   = src[H - 1][i + k] + M[1][0] * (temp2[H - 1][k] - src[H - 1][i + k]) + M[1][1] * (temp2[H - 2][k] - src[H - 1][i + k]) + M[1][2] * (temp2[H - 3][k] - src[H - 1][i + k]);
            temp2Hp1[k] = src[H - 1][i + k] + M[2][0] * (temp2[H - 1][k] - src[H - 1][i + k]) + M[2][1] * (temp2[H - 2][k] - src[H - 1][i + k]) + M[2][2] * (temp2[H - 3][k] - src[H - 1][i + k]);
        }

        for (int k = 0; k < numcols; k++) {
            dst[H - 1][i + k] *= temp2[H - 1][k] = temp2Hm1[k];
            dst[H - 2][i + k] *= temp2[H - 2][k] = B * temp2[H - 2][k] + b1 * temp2[H - 1][k] + b2 * temp2H[k] + b3 * temp2Hp1[k];
            dst[H - 3][i + k] *= temp2[H - 3][k] = B * temp2[H - 3][k] + b1 * temp2[H - 2][k] + b2 * temp2[H - 1][k] + b3 * temp2H[k];
        }

        for (int j = H - 4; j >= 0; j--) {
            for (int k = 0; k < numcols; k++) {auto
                dst[j][i + k] *= (temp2[j][k] = B * temp2[j][k] + b1 * temp2[j + 1][k] + b2 * temp2[j + 2][k] + b3 * temp2[j + 3][k]);
            }
        }
    }auto

#ifdef _OPENMP
    #pragma omp single
#endif

    // process remaining columns
    for (int i = W - (W % numcols); i < W; i++) {
        temp2[0][0] = B * src[0][i] + b1 * src[0][i] + b2 * src[0][i] + b3 * src[0][i];
        temp2[1][0] = B * src[1][i] + b1 * temp2[0][0]  + b2 * src[0][i] + b3 * src[0][i];
        temp2[2][0] = B * src[2][i] + b1 * temp2[1][0]  + b2 * temp2[0][0]  + b3 * src[0][i];

        for (int j = 3; j < H; j++) {
            temp2[j][0] = B * src[j][i] + b1 * temp2[j - 1][0] + b2 * temp2[j - 2][0] + b3 * temp2[j - 3][0];
        }

        double temp2Hm1 = src[H - 1][i] + M[0][0] * (temp2[H - 1][0] - src[H - 1][i]) + M[0][1] * (temp2[H - 2][0] - src[H - 1][i]) + M[0][2] * (temp2[H - 3][0] - src[H - 1][i]);
        double temp2H   = src[H - 1][i] + M[1][0] * (temp2[H - 1][0] - src[H - 1][i]) + M[1][1] * (temp2[H - 2][0] - src[H - 1][i]) + M[1][2] * (temp2[H - 3][0] - src[H - 1][i]);
        double temp2Hp1 = src[H - 1][i] + M[2][0] * (temp2[H - 1][0] - src[H - 1][i]) + M[2][1] * (temp2[H - 2][0] - src[H - 1][i]) + M[2][2] * (temp2[H - 3][0] - src[H - 1][i]);

        dst[H - 1][i] *= temp2[H - 1][0] = temp2Hm1;
        dst[H - 2][i] *= temp2[H - 2][0] = B * temp2[H - 2][0] + b1 * temp2[H - 1][0] + b2 * temp2H + b3 * temp2Hp1;
        dst[H - 3][i] *= temp2[H - 3][0] = B * temp2[H - 3][0] + b1 * temp2[H - 2][0] + b2 * temp2[H - 1][0] + b3 * temp2H;

        for (int j = H - 4; j >= 0; j--) {
            dst[j][i] *= (temp2[j][0] = B * temp2[j][0] + b1 * temp2[j + 1][0] + b2 * temp2[j + 2][0] + b3 * temp2[j + 3][0]);
        }
    }
}
#endif

template <class T> void reprocess(T** RESTRICT src, T** RESTRICT dst, const int W, const int H, const double sigma,  eGaussType gausstype,  reprocess_data* data, T** buffer2 = nullptr, float damping = 0.0f) {
 fprintf(stderr, "Checkpoint reprocess Rex\n"); fflush(stderr);
    static constexpr auto GAUSS_SKIP = 0.25;
    static constexpr auto GAUSS_3X3_LIMIT = 0.6;
    static constexpr auto GAUSS_DOUBLE = 25.0;
    /*Mimic decision process of GaussianBlurImpl below, ignoring the possible existence of buffer (note that it isn't a parameter, unlike with the original version below) since if that had been present, the OpenCL void OpenCLgauss3x3_ALL would not havd been triggered in the first place. 
We can also remove the < GAUSS_SKIP case since that will have been dealt with in the initial Impl filter */

    if ( (sigma > GAUSS_SKIP) && (sigma < GAUSS_3X3_LIMIT) ) {
            if(src != dst) {
                // If src != dst we can take the fast way
                // compute 3x3 kernel values
                double c0 = 1.0;
                double c1 = exp( -0.5 * (rtengine::SQR(1.0 / sigma)) );
                double c2 = exp( -rtengine::SQR(1.0 / sigma) );

                // normalize kernel values
                double sum = c0 + 4.0 * (c1 + c2);
                c0 /= sum;
                c1 /= sum;
                c2 /= sum;
                // compute kernel values for border pixels
                double b1 = exp (-1.0 / (2.0 * sigma * sigma));
                double bsum = 2.0 * b1 + 1.0;
                b1 /= bsum;
                double b0 = 1.0 / bsum;
	        fprintf(stderr, "Checkpoint reprocess Cloister\n"); fflush(stderr);
		
		*data =  reprocess_data{.c0 = c0, .c1 = c1, .c2 = c2, .b0 = b0, .b1 = b1};
		return;
		/* OpenCLgauss3x3_ALL will take the b and c double values and feed them into the appropriate
		kernels, depending on GaussType. */

            } else {
                // compute kernel values for separated 3x3 gaussian blur
                double c1 = exp (-1.0 / (2.0 * sigma * sigma));
                double csum = 2.0 * c1 + 1.0;
                c1 /= csum;
                double c0 = 1.0 / csum;
                gaussHorizontal3<T> (src, dst, W, H, c0, c1);
                gaussVertical3<T>   (dst, dst, W, H, c0, c1);

		
		data = NULL;
		return;
		
            }
        }
#ifdef __SSE2__

            if (sigma < GAUSS_DOUBLE) {
	       fprintf(stderr, "Checkpoint reprocess Double\n"); fflush(stderr);
                switch (gausstype) {
                case GAUSS_MULT : {
                    gaussHorizontalSse<T> (src, src, W, H, sigma);
                    gaussVerticalSsemult<T> (src, dst, W, H, sigma);
                    data = NULL;
		    return;
                }

                case GAUSS_DIV : {
                    gaussHorizontalSse<T> (src, dst, W, H, sigma);
                    gaussVerticalSsediv<T> (dst, dst, buffer2, W, H, sigma);
                    data = NULL;
		    return;
                }

                case GAUSS_STANDARD : {
                    gaussHorizontalSse<T> (src, dst, W, H, sigma);
                    gaussVerticalSse<T> (dst, dst, W, H, sigma);
                    data = NULL;
		    return;
                }
                }
            } else { // large sigma only with double precision
	       fprintf(stderr, "Checkpoint reprocess Large\n"); fflush(stderr);
                gaussHorizontal<T> (src, dst, W, H, sigma);
                gaussVertical<T>   (dst, dst, W, H, sigma);
		data = NULL;
		return;
            }

#else
      else {

            if (sigma < GAUSS_DOUBLE) {
                switch (gausstype) {
                case GAUSS_MULT : {
                    gaussHorizontal<T> (src, src, W, H, sigma);
                    gaussVerticalmult<T> (src, dst, W, H, sigma);
                    data = NULL;
		    return;
                }

                case GAUSS_DIV : {
                    gaussHorizontal<T> (src, dst, W, H, sigma);
                    gaussVerticaldiv<T> (dst, dst, buffer2, W, H, sigma);
                    data = NULL;
		    return;
                }

                case GAUSS_STANDARD : {
                    gaussHorizontal<T> (src, dst, W, H, sigma);
                    gaussVertical<T> (dst, dst, W, H, sigma);
                    data = NULL;
		    return;
                }
		}
            } else { // large sigma only with double precision  
                gaussHorizontal<T> (src, dst, W, H, sigma);
                gaussVertical<T>   (dst, dst, W, H, sigma);
	        data = NULL;
		return;
            }

        }
#endif
}

void replace_floats_and_doubles(double* b0, double* b1,
				double* c0, double* c1, double* c2,
				reprocess_data* b_and_c) {
	        if (b_and_c != NULL) {
		  *b0 = (float)b_and_c->b0; *b1 = b_and_c->b1;
		  *c0 = (float)b_and_c->c0; 
		  *c1 = (float)b_and_c->c1; 
		  *c2 = (float)b_and_c->c2;
      }
}

void reset_constants(cl_kernel kernel, double* b0_POINTER, double* b1_POINTER,
				 double* c0_POINTER, double* c1_POINTER, double* c2_POINTER) {
  int error_code;
     error_code = clSetKernelArg(kernel, 6, sizeof(cl_double), (void *)c0_POINTER);
     error_code = clSetKernelArg(kernel, 7, sizeof(cl_double), (void *)c1_POINTER);
     error_code = clSetKernelArg(kernel, 8, sizeof(cl_double), (void *)c2_POINTER);
}

  template<class T> void OpenCLgauss3x3_ALL (OpenCL_helper *helper, int iterations, eGaussType gausstype, T** RESTRICT src, T** RESTRICT dst, T** RESTRICT divBuffer, const int W, const int H,  double c0,  double c1,  double c2,  double b0,  double b1, float sigma, float damping)
{
    //turn array of arrays (the pointer to pointer(s)) into 2D array, excluding the outer rows and columns for newsrc,but  keeping them for oldsrc. Also creating dst for writing.
    float* newsrc = (float*) malloc( (W - 2) * (H - 2) * sizeof(float) ); //allocate the memory for the new src
    float* newdst = (float*) malloc( (W - 2) * (H - 2) * sizeof(float) ); //allocate the memory for the new dst, this will be pinned/mapped
    float* div  = (float*) malloc( W * H * sizeof(float) ); //allocate the memory for div
    
    float* oldsrc = (float*) malloc( W * H * sizeof(float) ); //allocate the memory for the old src, this will be pinned/mapped and will be manilated by the CPU
    float* olddst = (float*) malloc( W * H * sizeof(float) ); //allocate the memory for the old dst, this will be pinned/mapped
   
    int* X = (int*)malloc( (W - 2) * (H - 2) * sizeof(int));
    int* Y = (int*)malloc( (W - 2) * (H - 2) * sizeof(int));
    for (int i = 0; i < H; i++)
	{
	  for (int j = 0; j < W; j++)
	    {
	      oldsrc[i*W + j] = src[i][j];
	      olddst[i*W + j] = dst[i][j];
	      div[i*W + j] = divBuffer[i][j];
	    }
	}
    
    for (int i = 1; i < (H - 1); i++)
	{
	  int newi = i - 1;
	  for (int j = 1; j < (W - 1); j++)
	    {
	      int newW = W - 2;
	      int newj = j - 1;
	      newsrc[newi*newW + newj] = src[i][j]; //copy src to newsrc
	      newdst[newi*newW + newj] = dst[i][j]; //copy dst to newdst
	      
	      X[newi*newW + newj] = j;
	      Y[newi*newW + newj] = i;
	    }
	}

    fprintf(stderr, "src div pixel 1,1 is %f\n", src[1][1]);
    fprintf(stderr, "newsrc div pixel 1,1 is %f\n", newsrc[0]);
    fprintf(stderr, "Checkpoint div 3x reached\n");
    fflush(stderr);

    cl_kernel divkernel, standardkernel, multkernel;

    fprintf(stderr, "Checkpoint div Charlie reached\n"); fflush(stderr);

    standardkernel = helper->reuse_or_create_kernel(kernel_tag::gauss3x3std, "gauss_3x3_inner.cl", "gauss_3x3_inner");
    divkernel = helper->reuse_or_create_kernel(kernel_tag::gauss3x3div, "gauss_3x3_div_inner.cl", "gauss_3x3_div_inner");
    multkernel = helper->reuse_or_create_kernel(kernel_tag::gauss3x3mult, "gauss_3x3_mult_inner.cl", "gauss_3x3_mult_inner");
    cl_kernel dampingkernel = helper->reuse_or_create_kernel(kernel_tag::damping, "gauss_damping.cl", "gauss_damping");
    
    fprintf(stderr, "Checkpoint div Maha reached\n"); fflush(stderr);
    int Wi = W - 2; int Hi = H - 2; int error_code;

    cl_mem oldsrc_mem_obj = helper->reuse_or_create_buffer(helper->oldsrc_, W, H, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE);
     cl_mem olddst_mem_obj = helper->reuse_or_create_buffer(helper->olddst_, W, H, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE);
     cl_mem div_mem_obj = helper->reuse_or_create_buffer(helper->div_, W, H, CL_MEM_READ_WRITE);


    cl_mem ret_mem_obj = helper->reuse_or_create_buffer(helper->gaussret_, Wi, Hi, CL_MEM_READ_WRITE);
     cl_mem X_mem_obj = helper->reuse_or_create_buffer(helper->X_, Wi, Hi, CL_MEM_READ_ONLY);
     cl_mem Y_mem_obj = helper->reuse_or_create_buffer(helper->Y_, Wi, Hi, CL_MEM_READ_ONLY);

     /*error_code = clEnqueueWriteBuffer(helper->command_queue, oldsrc_mem_obj, CL_TRUE, 0, W*H*sizeof(float), oldsrc, 0, NULL, NULL); */
     float* mapped = (float*)clEnqueueMapBuffer(helper->command_queue, oldsrc_mem_obj, CL_TRUE, CL_MAP_WRITE, 0,  W*H*sizeof(float), NULL, NULL, NULL, &error_code);
     clFinish(helper->command_queue);
     fprintf(stderr, "first mapping is %d", error_code); fflush(stderr);
     memcpy(mapped, oldsrc, W*H*sizeof(float));

     float* dstmapped =  (float*)clEnqueueMapBuffer(helper->command_queue, olddst_mem_obj, CL_TRUE, CL_MAP_WRITE, 0,  W*H*sizeof(float), NULL, NULL, NULL, &error_code);
      clFinish(helper->command_queue);
       fprintf(stderr, "second  mapping is %d", error_code); fflush(stderr);
      memcpy(dstmapped, olddst, W*H*sizeof(float));
      
     error_code = clEnqueueWriteBuffer(helper->command_queue, ret_mem_obj, CL_TRUE, 0, Wi*Hi*sizeof(float), newdst, 0, NULL, NULL);   fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
      error_code = clEnqueueWriteBuffer(helper->command_queue, X_mem_obj, CL_TRUE, 0, Wi*Hi*sizeof(int), X, 0, NULL, NULL);
  fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
      error_code = clEnqueueWriteBuffer(helper->command_queue, Y_mem_obj, CL_TRUE, 0, Wi*Hi*sizeof(int), Y, 0, NULL, NULL);
  fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);
      error_code = clEnqueueWriteBuffer(helper->command_queue, div_mem_obj, CL_TRUE, 0, W*H*sizeof(float), div, 0, NULL, NULL);
        fprintf(stderr, "OpenCL Error code (0 is success):%d\n", error_code);  fflush(stderr);

     error_code = clSetKernelArg(standardkernel, 0, sizeof(cl_mem), (void *)&ret_mem_obj);
     error_code = clSetKernelArg(standardkernel, 1, sizeof(cl_mem), (void *)&oldsrc_mem_obj);
     error_code = clSetKernelArg(standardkernel, 2, sizeof(cl_mem), (void *)&X_mem_obj);
     error_code = clSetKernelArg(standardkernel, 3, sizeof(cl_mem), (void *)&Y_mem_obj);
     error_code = clSetKernelArg(standardkernel, 4, sizeof(cl_int), (void *)&W);
     error_code = clSetKernelArg(standardkernel, 5, sizeof(cl_int), (void *)&H);
     error_code = clSetKernelArg(standardkernel, 6, sizeof(cl_double), (void *)&c0);
     error_code = clSetKernelArg(standardkernel, 7, sizeof(cl_double), (void *)&c1);
     error_code = clSetKernelArg(standardkernel, 8, sizeof(cl_double), (void *)&c2);

     error_code = clSetKernelArg(multkernel, 0, sizeof(cl_mem), (void *)&ret_mem_obj);
     error_code = clSetKernelArg(multkernel, 1, sizeof(cl_mem), (void *)&olddst_mem_obj);
     error_code = clSetKernelArg(multkernel, 2, sizeof(cl_mem), (void *)&X_mem_obj);
     error_code = clSetKernelArg(multkernel, 3, sizeof(cl_mem), (void *)&Y_mem_obj);
     error_code = clSetKernelArg(multkernel, 4, sizeof(cl_int), (void *)&W);
     error_code = clSetKernelArg(multkernel, 5, sizeof(cl_int), (void *)&H);
     error_code = clSetKernelArg(multkernel, 6, sizeof(cl_double), (void *)&c0);
     error_code = clSetKernelArg(multkernel, 7, sizeof(cl_double), (void *)&c1);
     error_code = clSetKernelArg(multkernel, 8, sizeof(cl_double), (void *)&c2);
     error_code = clSetKernelArg(multkernel, 9, sizeof(cl_mem), (void *)&oldsrc_mem_obj);
     
     error_code = clSetKernelArg(divkernel, 0, sizeof(cl_mem), (void *)&ret_mem_obj);
     error_code = clSetKernelArg(divkernel, 1, sizeof(cl_mem), (void *)&oldsrc_mem_obj);
     error_code = clSetKernelArg(divkernel, 2, sizeof(cl_mem), (void *)&X_mem_obj);
     error_code = clSetKernelArg(divkernel, 3, sizeof(cl_mem), (void *)&Y_mem_obj);
     error_code = clSetKernelArg(divkernel, 4, sizeof(cl_int), (void *)&W);
     error_code = clSetKernelArg(divkernel, 5, sizeof(cl_int), (void *)&H);
     error_code = clSetKernelArg(divkernel, 6, sizeof(cl_double), (void *)&c0);
     error_code = clSetKernelArg(divkernel, 7, sizeof(cl_double), (void *)&c1);
     error_code = clSetKernelArg(divkernel, 8, sizeof(cl_double), (void *)&c2);
     error_code = clSetKernelArg(divkernel, 9, sizeof(cl_mem), (void *)&div_mem_obj);
     error_code = clSetKernelArg(divkernel, 10, sizeof(cl_mem), (void *)&olddst_mem_obj);

     error_code = clSetKernelArg(dampingkernel, 0, sizeof(cl_mem), (void *)&olddst_mem_obj);
     error_code = clSetKernelArg(dampingkernel, 1, sizeof(cl_mem), (void *)&div_mem_obj);
     error_code = clSetKernelArg(dampingkernel, 2, sizeof(cl_int), (void *)&W);
     error_code = clSetKernelArg(dampingkernel, 3, sizeof(cl_int), (void *)&H);
      error_code = clSetKernelArg(dampingkernel, 4, sizeof(cl_float), (void *)&damping);
     
     
    fprintf(stderr, "Checkpoint div 4x reached\n");
    fflush(stderr);

      
     size_t global_item_size = (H-2)*(W-2); 

    clock_t diff, diff2;
    clock_t start = clock();
    float *intermediate = (float*)malloc(Wi*Hi*sizeof(float));

    /*error_code =  clEnqueueUnmapMemObject(helper->command_queue, oldsrc_mem_obj, mapped, NULL, NULL, NULL);
      fprintf(stderr, "first UNmapping is %d", error_code); fflush(stderr);
       clFinish(helper->command_queue);
   error_code =  clEnqueueUnmapMemObject(helper->command_queue, olddst_mem_obj, dstmapped, NULL, NULL, NULL);
     fprintf(stderr, "second UNmapping is %d", error_code); fflush(stderr);
     clFinish(helper->command_queue); */

      
      cl_event ndevent0, ndevent; /*allows us to wait for events, so gauss operations are performed in sequence*/
      /* error_code = clEnqueueNDRangeKernel(helper->command_queue, divkernel, 1, NULL, &global_item_size, NULL, 0, NULL, &ndevent);
	 fprintf(stderr, "/nGPU enqueue ndrange kernel error code is %d", error_code); */

      reprocess_data* b_and_c = (reprocess_data*)malloc(sizeof(reprocess_data));
      
      /* */

    fprintf(stderr, "Beginning iteration cycles\n"); fflush(stderr);

    constexpr int sampleJ = 10;
    constexpr int sampleI = 10;
     
    for (int u = 0; u < iterations; u++) {
      
      fprintf(stderr, "New iteration cycle: %d\n", u); fflush(stderr);
      fprintf(stderr, "gpu src 10,10 is %f \n", mapped[10*W + 10]); fflush(stderr);
      fprintf(stderr, "gpu DST 10,10 is %f \n", dstmapped[10*W + 10]); fflush(stderr);
      
      if (damping == 0.0f) {
	fprintf(stderr, "Checkpoint Div"); fflush(stderr);
	reprocess<float>(src, dst, W,  H, sigma, GAUSS_DIV, b_and_c, divBuffer);
	if (b_and_c != NULL) {

     
	replace_floats_and_doubles(&b0, &b1, &c0, &c1, &c2, b_and_c);
	fprintf(stderr, "On gpu side: c0-%f, c1-%f, c2-%f and b0-%f, b1-%f\n", c0, c1, c2, b0, b1); fflush(stderr);
 
	reset_constants(divkernel, &b0, &b1, &c0, &c1, &c2);
     
	div_firstrow_and_firstcolumn(mapped, dstmapped, divBuffer, W, H, b0, b1);
	
	error_code =  clEnqueueUnmapMemObject(helper->command_queue, oldsrc_mem_obj, mapped, NULL, NULL, &ndevent0);
        error_code =  clEnqueueUnmapMemObject(helper->command_queue, olddst_mem_obj, dstmapped, 1, &ndevent0, &ndevent);
	
	error_code = clEnqueueNDRangeKernel(helper->command_queue, divkernel, 1, NULL, &global_item_size, NULL, 1, &ndevent, &ndevent);
	      mapped =  (float*)clEnqueueMapBuffer(helper->command_queue, oldsrc_mem_obj, CL_TRUE, CL_MAP_WRITE, 0,  W*H*sizeof(float), NULL, NULL, NULL, &error_code);
	      dstmapped =  (float*)clEnqueueMapBuffer(helper->command_queue, olddst_mem_obj, CL_TRUE, CL_MAP_WRITE, 0,  W*H*sizeof(float), NULL, NULL, NULL, &error_code);
	 
	 div_lastcolumn_and_lastrow(mapped, dstmapped, divBuffer, W,  H, b0, b1);
	 fprintf(stderr, "Checkpoint Memphis\n"); fflush(stderr);
	} 	 fprintf(stderr, "No. %d: Post div gpu %d,%d is %f \n", u, sampleI, sampleJ, dstmapped[sampleI*W + sampleJ]); fflush(stderr);
      }



      
      else if (damping > 0.0f)
	{
	 fprintf(stderr, "Checkpoint Standard\n"); fflush(stderr);
	reprocess<float>(src, dst, W,  H, sigma, GAUSS_STANDARD, b_and_c);
	if (b_and_c != NULL) {
	replace_floats_and_doubles(&b0, &b1, &c0, &c1, &c2, b_and_c);

	fprintf(stderr, "On gpu side: c0-%f, c1-%f, c2-%f and b0-%f, b1-%f\n", c0, c1, c2, b0, b1); fflush(stderr);

        reset_constants(standardkernel, &b0, &b1, &c0, &c1, &c2);
        //standard gauss3x3

        standard_firstrow_and_firstcolumn(mapped, dstmapped, W, H, b0, b1);
	

	error_code =  clEnqueueUnmapMemObject(helper->command_queue, oldsrc_mem_obj, mapped, NULL, NULL, NULL);
        error_code =  clEnqueueUnmapMemObject(helper->command_queue, olddst_mem_obj, dstmapped, NULL, NULL, NULL);
     
        error_code = clEnqueueNDRangeKernel(helper->command_queue, standardkernel, 1, NULL, &global_item_size, NULL, 0, NULL, &ndevent0);
   
        mapped =  (float*)clEnqueueMapBuffer(helper->command_queue, oldsrc_mem_obj, CL_TRUE, CL_MAP_WRITE, 0,  W*H*sizeof(float), NULL, NULL, NULL, &error_code);
	dstmapped =  (float*)clEnqueueMapBuffer(helper->command_queue, olddst_mem_obj, CL_TRUE, CL_MAP_WRITE, 0,  W*H*sizeof(float), NULL, NULL, NULL, &error_code);

	standard_lastcolumn_and_lastrow(mapped, dstmapped, W, H, b0, b1);
	
	}
	/****damping****/
	error_code =  clEnqueueUnmapMemObject(helper->command_queue, olddst_mem_obj, dstmapped, NULL, NULL, NULL);
        error_code = clEnqueueNDRangeKernel(helper->command_queue, dampingkernel, 1, NULL, &global_item_size, NULL, 1, &ndevent0, &ndevent);
	dstmapped =  (float*)clEnqueueMapBuffer(helper->command_queue, olddst_mem_obj, CL_TRUE, CL_MAP_WRITE, 0,  W*H*sizeof(float), NULL, NULL, NULL, &error_code);
	fprintf(stderr, "No. %d: gpu damping resullt %d,%d is %f \n", u, sampleI, sampleJ, dstmapped[sampleI*W + sampleJ]);
        }



      fprintf(stderr, "Checkpoint Mult\n"); fflush(stderr);
     reprocess<float>(dst, src, W,  H, sigma, GAUSS_MULT, b_and_c);
     
     	if (b_and_c != NULL) {
     replace_floats_and_doubles(&b0, &b1, &c0, &c1, &c2, b_and_c);
     reset_constants(multkernel, &b0, &b1, &c0, &c1, &c2);

     mult_firstrow_and_firstcolumn(dstmapped, mapped, W, H, b0, b1);

     error_code =  clEnqueueUnmapMemObject(helper->command_queue, olddst_mem_obj, dstmapped, NULL, NULL, NULL);
     error_code =  clEnqueueUnmapMemObject(helper->command_queue, oldsrc_mem_obj, mapped, NULL, NULL, NULL);
     
     error_code = clEnqueueNDRangeKernel(helper->command_queue, multkernel, 1, NULL, &global_item_size, NULL, 0, NULL, &ndevent);

       dstmapped =  (float*)clEnqueueMapBuffer(helper->command_queue, olddst_mem_obj, CL_TRUE, CL_MAP_WRITE, 0,  W*H*sizeof(float), NULL, NULL, NULL, &error_code);
      mapped =  (float*)clEnqueueMapBuffer(helper->command_queue, oldsrc_mem_obj, CL_TRUE, CL_MAP_WRITE, 0,  W*H*sizeof(float), NULL, NULL, NULL, &error_code);
  
      mult_lastcolumn_and_lastrow(dstmapped, mapped, W, H, b0, b1);
	 
	} fprintf(stderr, "No. %d: Post mult gpu tmpI/src %d,%d is %f \n", u, sampleI, sampleJ, mapped[sampleI*W + sampleJ]); fflush(stderr);
    }


    diff = clock() - start;
     double msec = (double)diff * 1000.0 / (double)CLOCKS_PER_SEC;
     fprintf(stderr, "OpenCL took %f to calculate", msec);
     fflush(stderr);

     free(X); free(Y);

     rtengine::JaggedArray<float> GPU(W, H);

      // CPU
        dst[0][0] = rtengine::max(divBuffer[0][0] / (src[0][0] > 0.f ? src[0][0] : 1.f), 0.f);

        for (int j = 1; j < W - 1; j++)
        {
            float tmp = (b1 * (src[0][j - 1] + src[0][j + 1]) + b0 * src[0][j]);
            dst[0][j] = rtengine::max(divBuffer[0][j] / (tmp > 0.f ? tmp : 1.f), 0.f);
        }

        dst[0][W - 1] = rtengine::max(divBuffer[0][W - 1] / (src[0][W - 1] > 0.f ? src[0][W - 1] : 1.f), 0.f);
    
      for (int i = 1; i < H - 1; i++) {
	float tmp = (b1 * (src[i - 1][0] + src[i + 1][0]) + b0 * src[i][0]);
        dst[i][0] = rtengine::max(divBuffer[i][0] / (tmp > 0.f ? tmp : 1.f), 0.f);

        for (int j = 1; j < W - 1; j++) {
	  tmp = (c2 * (src[i - 1][j - 1] + src[i - 1][j + 1] + src[i + 1][j - 1] + src[i + 1][j + 1]) + c1 * (src[i - 1][j] + src[i][j - 1] + src[i][j + 1] + src[i + 1][j]) + c0 * src[i][j]);
	  dst[i][j] = rtengine::max(divBuffer[i][j] / (tmp > 0.f ? tmp : 1.f), 0.f);
        }
	
	tmp = (b1 * (src[i - 1][W - 1] + src[i + 1][W - 1]) + b0 * src[i][W - 1]);
        dst[i][W - 1] = rtengine::max(divBuffer[i][W - 1] / (tmp > 0.f ? tmp : 1.f), 0.f);

    }
      //last row
       dst[H - 1][0] = rtengine::max(divBuffer[H - 1][0] / (src[H - 1][0] > 0.f ? src[H - 1][0] : 1.f), 0.f);

        for (int j = 1; j < W - 1; j++) {
            float tmp = (b1 * (src[H - 1][j - 1] + src[H - 1][j + 1]) + b0 * src[H - 1][j]);
            dst[H - 1][j] = rtengine::max(divBuffer[H - 1][j] / (tmp > 0.f ? tmp : 1.f), 0.f);
        }

        dst[H - 1][W - 1] = rtengine::max(divBuffer[H - 1][W - 1] / (src[H - 1][W - 1] > 0.f ? src[H - 1][W - 1] : 1.f), 0.f);
   
      //error_code = clEnqueueUnmapMemObject(helper->command_queue, olddst_mem_obj, dstmapped, NULL, NULL, NULL);

       fprintf(stderr, "CPU div work says %d,%d is %f; ", 1, 1, dst[1][1]); fflush(stderr);
      fprintf(stderr, "CPU div work says %d,%d is %f; ", 10, 10, dst[10][10]); fflush(stderr);
      fprintf(stderr, "CPU div work says %d,%d is %f; ", 105, 234, dst[105][234]); fflush(stderr);
      
     //copy the memory back to the inner bit of the destination 2D array
      
      float tmp;
      
       for (int i = 0; i < H; i++)
	       {
	        for (int j = 0; j < W; j++)
	         {
		   GPU[i][j] = dstmapped[i*W + j]; 
	         }
             	}

    for (int i = 1; i < (H - 1); i++)
	{
	  int newi = i - 1;
	  for (int j = 1; j < (W - 1); j++)
	    {
	      int newW = W - 2;
	      int newj = j - 1;
	      //GPU[i][j] = intermediate[newi*newW + newj];
	    }
	  tmp = (b1 * (src[i - 1][W - 1] + src[i + 1][W - 1]) + b0 * src[i][W - 1]);
	  //  GPU[i][W - 1] = rtengine::max(divBuffer[i][W - 1] / (tmp > 0.f ? tmp : 1.f), 0.f);
	}
    fprintf(stderr, "Checkpoint div ab reached  \n");

    fprintf(stderr, "intermediate 0,0 is %f", intermediate[0]);
    fprintf(stderr, "W is %d, H is %d, GPU div work says %d,%d is %f; ",W, H, 10, 10, GPU[10][10]); fflush(stderr);
      fprintf(stderr, "GPU div work says %d,%d is %f; ", 105, 234, GPU[105][234]); fflush(stderr);

      char string[100];
      int v, h;
 int exit = 0;
 /* while (exit == 0) {
    scanf("%d %d %d", &v, &h, &exit);
    fprintf(stderr, "CPU div work says %d,%d is %f; ", v, h, dst[v][h]); fflush(stderr);
    fprintf(stderr, "GPU div work says %d,%d is %f; ", v, h, GPU[v][h]); fflush(stderr);
    }*/
 free(intermediate); free(newdst); free(newsrc); free(div); free(olddst); free(oldsrc); free(b_and_c);
}

 template<class T> void gaussianBlurImpl(OpenCL_helper* helper, int iterations, T** src, T** dst, const int W, const int H, const double sigma, T *buffer = nullptr, eGaussType gausstype = GAUSS_STANDARD, T** buffer2 = nullptr, float damping = 0.0f)
{
    static constexpr auto GAUSS_SKIP = 0.25;
    static constexpr auto GAUSS_3X3_LIMIT = 0.6;
    static constexpr auto GAUSS_DOUBLE = 25.0;

    if(buffer) {
      fprintf(stderr, "Checkpoint Quincy\n"); fflush(stderr);
        // special variant for very large sigma, currently only used by retinex algorithm
        // use iterated boxblur to approximate gaussian blur
        // Compute ideal averaging filter width and number of iterations
        int n = 1;
        double wIdeal = sqrt((12 * sigma * sigma) + 1);

        while(wIdeal > W || wIdeal > H) {
            n++;
            wIdeal = sqrt((12 * sigma * sigma / n) + 1);
        }

        if(n < 3) {
            n = 3;
            wIdeal = sqrt((12 * sigma * sigma / n) + 1);
        } else if(n > 6) {
            n = 6;
        }

        int wl = wIdeal;

        if(wl % 2 == 0) {
            wl--;
        }

        int wu = wl + 2;

        double mIdeal = (12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4);
        int m = round(mIdeal);

        int sizes[n];

        for(int i = 0; i < n; i++) {
            sizes[i] = ((i < m ? wl : wu) - 1) / 2;
        }

        rtengine::boxblur(src, dst, buffer, sizes[0], sizes[0], W, H);

        for(int i = 1; i < n; i++) {
            rtengine::boxblur(dst, dst, buffer, sizes[i], sizes[i], W, H);
        }
    } else {
        if (sigma < GAUSS_SKIP) {
	  fprintf(stderr, "Checkpoint Marine\n"); fflush(stderr);
            // don't perform filtering
            if (src != dst) {
                for(int i = 0; i < H; ++i) {
                    memcpy(dst[i], src[i], W * sizeof(T));
                }
            }
        } else if (sigma < GAUSS_3X3_LIMIT) {
            if(src != dst) {
                // If src != dst we can take the fast way
                // compute 3x3 kernel values
                double c0 = 1.0;
                double c1 = exp( -0.5 * (rtengine::SQR(1.0 / sigma)) );
                double c2 = exp( -rtengine::SQR(1.0 / sigma) );

                // normalize kernel values
                double sum = c0 + 4.0 * (c1 + c2);
                c0 /= sum;
                c1 /= sum;
                c2 /= sum;
                // compute kernel values for border pixels
                double b1 = exp (-1.0 / (2.0 * sigma * sigma));
                double bsum = 2.0 * b1 + 1.0;
                b1 /= bsum;
                double b0 = 1.0 / bsum;

                switch (gausstype) {
                case GAUSS_MULT     :
                    gauss3x3mult<T> (src, dst, W, H, c0, c1, c2, b0, b1);
                    break;

                case GAUSS_DIV      :
		  fprintf(stderr, "Engaging OpenCL gauss 3x3 div\n");
	            fflush(stderr);
		    if (helper!= NULL) 
		    OpenCLgauss3x3_ALL<T> (helper, iterations, gausstype, src, dst, buffer2, W, H, c0, c1, c2, b0, b1, sigma, damping); //sigma added for reprocessing
		    else { 	fprintf(stderr, "On CPU div side: c0-%f, c1-%f, c2-%f and b0-%f, b1-%f\n", c0, c1, c2, b0, b1); fflush(stderr);
		      fprintf(stderr, "earlier point CPU 10,10 is %f\n", src[10][10]); fflush(stderr);
		      gauss3x3div<T> (src, dst, buffer2, W, H, c0, c1, c2, b0, b1);
		      /* for (int i = 0; i<H; i++) {
			for (int j = 0; j<W; j++) {
			  dst[i][j] = buffer2[i][j];
			}
			}*/
		    }
                    break;

                case GAUSS_STANDARD :
		    fprintf(stderr, "Engaging OpenCL gauss 3x3 standard\n");
	            fflush(stderr);
		    if (helper!= NULL) 
                    OpenCLgauss3x3<T> (helper, src, dst, W, H, c0, c1, c2, b0, b1);
		    else gauss3x3<T> (src, dst, W, H, c0, c1, c2, b0, b1);
                    break;
                }
            } else {
                // compute kernel values for separated 3x3 gaussian blur
                double c1 = exp (-1.0 / (2.0 * sigma * sigma));
                double csum = 2.0 * c1 + 1.0;
                c1 /= csum;
                double c0 = 1.0 / csum;
                gaussHorizontal3<T> (src, dst, W, H, c0, c1);
                gaussVertical3<T>   (dst, dst, W, H, c0, c1);
            }
        } else {
#ifdef __SSE2__

            if (sigma < GAUSS_DOUBLE) {
                switch (gausstype) {
                case GAUSS_MULT : {
                    gaussHorizontalSse<T> (src, src, W, H, sigma);
                    gaussVerticalSsemult<T> (src, dst, W, H, sigma);
                    break;
                }

                case GAUSS_DIV : {
                    gaussHorizontalSse<T> (src, dst, W, H, sigma);
                    gaussVerticalSsediv<T> (dst, dst, buffer2, W, H, sigma);
                    break;
                }

                case GAUSS_STANDARD : {
                    gaussHorizontalSse<T> (src, dst, W, H, sigma);
                    gaussVerticalSse<T> (dst, dst, W, H, sigma);
                    break;
                }
                }
            } else { // large sigma only with double precision
                gaussHorizontal<T> (src, dst, W, H, sigma);
                gaussVertical<T>   (dst, dst, W, H, sigma);
            }

#else

            if (sigma < GAUSS_DOUBLE) {
                switch (gausstype) {
                case GAUSS_MULT : {
                    gaussHorizontal<T> (src, src, W, H, sigma);
                    gaussVerticalmult<T> (src, dst, W, H, sigma);
                    break;
                }

                case GAUSS_DIV : {
                    gaussHorizontal<T> (src, dst, W, H, sigma);
                    gaussVerticaldiv<T> (dst, dst, buffer2, W, H, sigma);
                    break;
                }

                case GAUSS_STANDARD : {
                    gaussHorizontal<T> (src, dst, W, H, sigma);
                    gaussVertical<T> (dst, dst, W, H, sigma);
                    break;
                }
                }
            } else { // large sigma only with double precision  
                gaussHorizontal<T> (src, dst, W, H, sigma);
                gaussVertical<T>   (dst, dst, W, H, sigma);
            }

#endif
        }
    }
}
}

void gaussianBlur(float** src, float** dst, const int W, const int H, const double sigma, float *buffer, eGaussType gausstype, float** buffer2)
{
  gaussianBlurImpl<float>(NULL, NULL, src, dst, W, H, sigma, buffer, gausstype, buffer2);
}

void OpenCLgaussianBlur(OpenCL_helper* helper, int iterations, float** src, float** dst, const int W, const int H, const double sigma, float *buffer, eGaussType gausstype, float** buffer2, float damping)
{
  gaussianBlurImpl<float>(helper, iterations, src, dst, W, H, sigma, buffer, gausstype, buffer2, damping);
}




