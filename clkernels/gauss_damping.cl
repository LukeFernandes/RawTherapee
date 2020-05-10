#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void gauss_damping( __global float *aI, __global const float *aO,  const int W, const int H, const float dampinFac) {
// Get the index of the current element to be processed
int index = get_global_id(0);
}

/* Transposing the (non SSE2) code for damping */

// Get the index of the current element to be processed
/*int index = get_global_id(0);

float I = aI(index);
float O = aO(index);
 printf("eeeeeeeeeeeeeeee");

if (O <= 0.f || I <= 0.f) {
  aI[i][j] = 4.0f//0.f;
         }
else {


   float U = (O * xlogf(I / O) - I + O) * dampingFac;
   U = fmin(U, 1.0f);
   U = U * U * U * U * (5.f - U * 4.f);
   aI[i][j] = 5.0f; //(O - I) / I * U + 1.f;

}

}

float xlogf(float d) {
  float x, x2, t, m;
  int e;

  e = ilogbp1f(d * 0.7071f);
  m = ldexpkf(d, -e);

  x = (m-1.0f) / (m+1.0f);
  x2 = x * x;

  t = 0.2371599674224853515625f;
  t = mlaf(t, x2, 0.285279005765914916992188f);
  t = mlaf(t, x2, 0.400005519390106201171875f);
  t = mlaf(t, x2, 0.666666567325592041015625f);
  t = mlaf(t, x2, 2.0f);

  x = x * t + 0.693147180559945286226764f * e;

  if (xisinff(d)) x = INFINITY;
  if (d < 0) x = NAN;
  if (d == 0) x = INFINITY;

  return x;
}

int xisinff(float x) { return x == INFINITY || x == -INFINITY; }

float mlaf(float x, float y, float z) {
  return x * y + z; }

int floatToRawIntBits(float d) {
  union {
    float f;
    int i;
  } tmp;
  tmp.f = d;
  return tmp.i;
}

int ilogbf1f(float d) {
  int m = d < 5.421010862427522E-20f;
  
  //d = m ? 1.8446744073709552E19f * d : d;
   if (m) d = 1.8446744073709552E19f * d;
  else d = d;
  int q = (floatToRawIntBits(d) >> 23) & 0xff;
 // q = m ? q - (64 + 0x7e) : q - 0x7e;
 if (m) q = q - (64 + 0x7e);
     else q = q - 0x7e;
  return q;
}

float intBitsToFloat(int i) {
  union {
    float f;
    int i;
  } tmp;
  tmp.i = i;
  return tmp.f;
}

float ldexpkf(float x, int q) {
  float u;
  int m;
  m = q >> 31;
  m = (((m + q) >> 6) - m) << 4;
  q = q - (m << 2);
  u = intBitsToFloat((int)(m + 0x7f) << 23);
  u = u * u;
  x = x * u * u;
  u = intBitsToFloat((int)(q + 0x7f) << 23);
  return x * u;
}
