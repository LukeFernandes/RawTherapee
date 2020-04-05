__kernel void intp_plus_version2(__global const float *blend, __global const float *tmpI, __global float *lum, __global const int *X, __global const int *Y, const double amount) {
 
    // Get the index of the current element to be processed
    int index = get_global_id(0);
    int currentY = Y[index]; int currentX = X[index];
    
    //from rt_math
    // calculate a * b + (1 - a) * c
    // following is valid:
    // intp(a, b+x, c+x) = intp(a, b, c) + x
    // intp(a, b*x, c*x) = intp(a, b, c) * x

    /*luminance[i][j] = intp(blend_jagged_array[i][j] * amount, max(tmpI[i][j], 0.0f), luminance[i][j]);*/

    double temp_term = blend[index] * amount;

    // max the tmpI
    float maxresult = fmax(tmpI[index], 0.0f);

     if (currentX == 900 && currentY == 900) {
       printf("\n INTP2: in gpu blend is %f", blend[index]);
       printf("\n INTP2: in gpu tmpI is %f", tmpI[index]);
       printf("\n INTP2: in gpu lum is %f", lum[index]);
       printf("\n INTP2: in gpu temp term is %f", temp_term);
     }
    
    // Do the operation *intp*
    lum[index] = temp_term * (maxresult - lum[index]) + lum[index];
    

}
