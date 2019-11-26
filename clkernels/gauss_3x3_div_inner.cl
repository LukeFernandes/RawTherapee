#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void gauss_3x3_div_inner(__global float *ret, __global const float *oldsrc, __global const int *X, __global const int *Y, const int W, const int H, const double c0, const double c1, const double c2, __global float *div, __global float *olddst) {

/* Because of the subsetting necessary for the inner algorithm, we use the buffer subset and refer to the original unmodified buffer as inputs. Note that the use of OpenCL sub buffers would not be possible here. */

// Get the index of the current element to be processed
int index = get_global_id(0);

int index_equiv = Y[index] * W + X[index];

/* Note that e.g. (index - W - 1) is equivalent to (j - 1) because we are subtracting one row's worth of width to get to the row above, i.e. to go one up the Y axis.*/

 float output = c2 * (oldsrc[index_equiv - W - 1] + oldsrc[index_equiv - W + 1] + oldsrc[index_equiv + W - 1] + oldsrc[index_equiv + W + 1]) + c1 * (oldsrc[index_equiv - W] + oldsrc[index_equiv - 1] + oldsrc[index_equiv + 1] + oldsrc[index_equiv + W]) + c0*oldsrc[index_equiv]; 

 //ret[index] = oldsrc[index_equiv];

float term = 1.f;
if (output > 0.f) term = output;
 float final = fmax( (div[index_equiv] / term), 0.f );
 ret[index] = final;
 olddst[index_equiv] = final; 

}
