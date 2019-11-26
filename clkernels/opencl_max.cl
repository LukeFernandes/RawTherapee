__kernel void opencl_max(__global const float *lum, __global float *ret) {
 
    // Get the index of the current element to be processed
    int index = get_global_id(0);
 
    // Do the operation
    ret[index] = fmax(0.0f, lum[index]);
}
