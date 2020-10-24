__kernel void intp_plus_version2(__global const float *blend, __global const float *tmpI, __global float *lum, const float amount) {
 
    // Get the index of the current element to be processed
    int index = get_global_id(0);
    
    /*from rt_math
    return a * (b - c) + c;
    also incorporating the max bit:
    luminance[i][j] = intp(blend_jagged_array[i][j] * amount, max(tmpI[i][j], 0.0f), luminance[i][j]);*/

    double temp_term = blend[index] * amount;

    // max the tmpI
    float maxresult = fmax(tmpI[index], 0.0f);
    // Do the operation *intp*
    lum[index] = temp_term * (maxresult - lum[index]) + lum[index];
    
    /* if (currentX == 900 && currentY == 900) {
       printf("\n INTP2: in gpu blend is %f", blend[index]);
       printf("\n INTP2: in gpu tmpI is %f", tmpI[index]);
       printf("\n INTP2: in gpu lum is %f", lum[index]);
       printf("\n INTP2: in gpu temp term is %f", temp_term);
       } */
    
}
