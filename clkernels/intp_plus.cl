float cmax(float f1, float f2);

__kernel void intp_plus(__global const float *blend, __global const float *lum, __global float *blur) {
 
    // Get the index of the current element to be processed
    int index = get_global_id(0);
    
    //from rt_math
    // calculate a * b + (1 - a) * c
    // following is valid:
    // intp(a, b+x, c+x) = intp(a, b, c) + x
    // intp(a, b*x, c*x) = intp(a, b, c) * x
    // return a * (b - c) + c;

    // max the blur
    float maxresult = fmax(blur[index], 0.0f);
    float max2 = cmax(blur[index], 0.0f);

    // Do the operation *intp*
    blur[index] =  blend[index] * (lum[index] - max2) + max2;
    //ret[index] = 5.0f; 
    //ret[index] = blend[index] * lum[index] + (1 - blend[index]) * maxresult;

}

float cmax(float f1, float f2) {

  if (f1 > f2) return f1;
  else return f2;

}
