#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void gauss_5x5_mult_whole( __global const float *oldsrc, __global const int *X, __global const int *Y,  const int W, const int H, const double b1, const double b2, const double b3, const double B, const double c00, __global float *div, __global float *olddst) {
 
// Get the index of the current element to be processed
int index = get_global_id(0);

int currentY = Y[index]; int currentX = X[index];

int index_equiv = currentY * W + currentX;
//printf("index_equiv is %d", index_equiv);

 int checkintX = 100; int checkintY = 100;

/*This section avoids conditioal branching*/

 // printf("borders done");
 
// Check if on the borders
 int topborder_check = (1 - clamp(Y[index], 0, 1)); 
 int top_but_one_border_check = (1 - clamp(Y[index] - 1, 0, 1));
 int leftborder_check = (1 - clamp(X[index], 0, 1)); 
 int left_but_one_border_check = (1 - clamp(X[index] - 1, 0, 1));
 int left_but_two_border_check = (1 - clamp(X[index] - 2, 0, 1)); 
 int rightborder_check = (1 - clamp((W - 1 - X[index]), 0, 1)); 
 int right_but_one_border_check = (1 - clamp((W - 1 - X[index] + 1), 0, 1));
 int bottomborder_check = (1 - clamp((H - 1 - Y[index]), 0, 1)); 
 int bottom_but_one_border_check = (1 - clamp((H - 1 - Y[index] + 1), 0, 1));

 //Preparatory values need to be set for first three columns and last three columns

 /*   temp2[0] = B * src[i][0] + b1 * src[i][0] + b2 * src[i][0] + b3 * src[i][0];
       temp2[1] = B * src[i][1] + b1 * temp2[0]  + b2 * src[i][0] + b3 * src[i][0];
       temp2[2] = B * src[i][2] + b1 * temp2[1]  + b2 * temp2[0]  + b3 * src[i][0]; */
 double temp =
   leftborder_check * ( B * src[index_equiv] + b1 * src[index_equiv] + b2 * src[index_equiv] + b3 * src[index_equiv] );
   left_but_one_border_check * ( B * src[index_equiv] + b1 * src[index_equiv] + b2 * src[index_equiv] + b3 * src[index_equiv] );
   left_but_two_border_check * (B * src[index_equiv] + b1 * src[index_equiv] + b2 * src[index_equiv] + b3 * src[index_equiv] );
 

   int innerpixels_check = (1 - top_but_one_border_check)*(1 - left_but_one_border_check)*(1 - right_but_one_border_check)*(1 - bottom_but_one_border_check)*(1 - topborder_check)*(1 - leftborder_check)*(1 - rightborder_check)*(1 - bottomborder_check); 

float inner_temp1 =  B * src[index_equiv] + b1 * temp2[j - 1] + b2 * temp2[j - 2] + b3 * temp2[j - 3];
  
  if ((currentX == checkintX) && (currentY == checkintY)) {

   
  }

  dst[index_equiv] = dst[index_equiv] * temp_pre_mult;

}

