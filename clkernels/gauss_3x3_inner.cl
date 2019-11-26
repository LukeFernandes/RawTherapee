#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void gauss_3x3_inner(__global float *ret, __global const float *oldsrc, __global const int *X, __global const int *Y, const int W, const int H, const double c0, const double c1, const double c2, __global float *olddst) {

/* Because of the subsetting necessary for the inner algorithm, we use the buffer subset and refer to the original unmodified buffer as inputs. Note that the use of OpenCL sub buffers would not be possible here. */

// Get the index of the current element to be processed
int index = get_global_id(0);

int index_equiv = Y[index] * W + X[index];

//first row excluding first and last column
 ret[index] = c2 * (oldsrc[index_equiv - W - 1] + oldsrc[index_equiv - W + 1] + oldsrc[index_equiv + W - 1] + oldsrc[index_equiv + W + 1]) + c1 * (oldsrc[index_equiv - W] + oldsrc[index_equiv - 1] + oldsrc[index_equiv + 1] + oldsrc[index_equiv + W]) + c0*oldsrc[index_equiv]; 

olddst[index_equiv] = ret[index];

}
