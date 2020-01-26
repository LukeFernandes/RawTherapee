__kernel void intp_plus(__global const float *blend, __global const float *lum, __global const float *blur, __global float *ret) {
 
    // Get the index of the current element to be processed
    int index = get_global_id(0);
    
    //from rt_math
    // calculate a * b + (1 - a) * c
    // following is valid:
    // intp(a, b+x, c+x) = intp(a, b, c) + x
    // intp(a, b*x, c*x) = intp(a, b, c) * x

    // max the blur
    float maxresult = fmax(blur[index], 0.0f);

    // Do the operation *intp*
    ret[index] = blend[index] * (lum[index] - maxresult) + maxresult;
    //ret[index] = 5.0f;
    //ret[index] = blend[index] * lum[index] + (1 - blend[index]) * maxresult;

}
