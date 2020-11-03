__kernel void intp_plus_version3(__global const float *blend, __global float *lum, __global const float *blur) {
 
    // Get the index of the current element to be processed
    int index = get_global_id(0);    
    //from rt_math
    //return a * (b - c) + c;

    // max the blur
    float maxresult = fmax(blur[index], 0.0f);

    // Do the operation *intp*
    lum[index] =  blend[index] * (lum[index] - maxresult) + maxresult;
    /*if (currentX == 900 && currentY == 900) {
    printf("\n in gpu blend is %f", blend[index]);
    printf("\n in gpu lum is %f", lum[index]);
     printf("\n in gpu blur is %f", blur[index]);
    printf("\n in gpu subtraction is %f", (lum[index] - maxresult));
    printf("\n in gpu maxresult is %f", maxresult); } */
    
}
