__kernel void OpenCl_max(_global const int *lum, __global int *ret) {
 
    // Get the index of the current element to be processed
    int index = get_global_id(0);
 
    // Do the operation
    ret[index] = fmax(0.0f, lum[index]);
}