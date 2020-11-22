//#define M(r, c)  (M[(r)*4 + (c)])

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void gauss_horizontal( __global const float *src, __global const int *X, __global const int *Y,  const int W, const int H, const double b1, const double b2, const double b3, const double B, __global const double16 *Maddr, __global double *intermediate, __global float *dst) {

  double16 M = *Maddr;
int totalpix = (W*H)-1;;
// Get the index of the current element to be processed
int index = get_global_id(0);

int currentY = Y[index]; int currentX = X[index];

int index_equiv = currentY * W + currentX;
//printf("index_equiv is %d", index_equiv);

 int checkintX = 100; int checkintY = 100;

/*This section avoids conditional branching*/

 // printf("borders done");
 
// Check if on the borders
 int leftborder_check = (1 - clamp(X[index], 0, 1)); 
 int left_but_one_border_check = (1 - clamp(X[index] - 1, 0, 1));
 int left_but_two_border_check = (1 - clamp(X[index] - 2, 0, 1)); 
 int rightborder_check = (1 - clamp((W - 1 - X[index]), 0, 1)); 
 int right_but_one_border_check = (1 - clamp((W - 1 - (X[index] + 1) ), 0, 1));
  int right_but_two_border_check = (1 - clamp((W - 1 - (X[index] + 2) ), 0, 1));
 int right_but_three_border_check = (1 - clamp((W - 1 - (X[index] + 3) ), 0, 1));

 //Preparatory values need to be set for first three columns and last three columns

 /*   temp2[0] = B * src[i][0] + b1 * src[i][0] + b2 * src[i][0] + b3 * src[i][0];
       temp2[1] = B * src[i][1] + b1 * temp2[0]  + b2 * src[i][0] + b3 * src[i][0];
       temp2[2] = B * src[i][2] + b1 * temp2[1]  + b2 * temp2[0]  + b3 * src[i][0]; */
 double left =
   leftborder_check * ( B * src[index_equiv] + b1 * src[index_equiv] + b2 * src[index_equiv] + b3 * src[index_equiv] );
   left_but_one_border_check * ( B * src[index_equiv] + b1 * src[index_equiv] + b2 * src[index_equiv] + b3 * src[index_equiv] );
   left_but_two_border_check * (B * src[index_equiv] + b1 * src[index_equiv] + b2 * src[index_equiv] + b3 * src[index_equiv] );
 

   // int innerpixels_check = (1 - top_but_one_border_check)*(1 - left_but_one_border_check)*(1 - right_but_one_border_check)*(1 - bottom_but_one_border_check)*(1 - topborder_check)*(1 - leftborder_check)*(1 - rightborder_check)*(1 - bottomborder_check);

   int leftpixels_check = 1 - (1 - leftborder_check)*(1 - left_but_one_border_check)*(1 - left_but_two_border_check);
   int rightwardpixels_check = (1 - leftborder_check)*(1 - left_but_one_border_check)*(1 - left_but_two_border_check);

   int rightpixels_check = 1 - (1 - rightborder_check)*(1 - right_but_one_border_check)*(1 - right_but_two_border_check);
   int leftwardpixels_check = (1 - rightborder_check)*(1 - right_but_one_border_check)*(1 - right_but_two_border_check);

   intermediate[index_equiv] = rightwardpixels_check * ( B * src[index_equiv] + b1 * src[index_equiv - 1] + b2 * src[index_equiv - 2] + b3 * src[index_equiv - 3] )
     + leftpixels_check * left;

     if ((currentX == checkintX) && (currentY == checkintY)) {
       printf("initial pass results in %f",  intermediate[index_equiv]);				   
  }

barrier(CLK_GLOBAL_MEM_FENCE);

 double temp2Wm1 = src[currentY*W + (W-1)] + 

   M.s0 * (intermediate[currentY*W + (W-1)] - src[currentY*W + (W-1)])

   + M.s1 * (intermediate[currentY*W + (W-2)] - src[currentY*W + (W-1)])

   + M.s2 * (intermediate[currentY*W + (W-3)] - src[currentY*W + (W-1)]);

  double temp2W = src[currentY*W + (W-1)] + 

    M.s4 * (intermediate[currentY*W + (W-1)] - src[currentY*W + (W-1)])

    + M.s5 * (intermediate[currentY*W + (W-2)] - src[currentY*W + (W-1)])

    + M.s6 * (intermediate[currentY*W + (W-3)] - src[currentY*W + (W-1)]);

   double temp2Wp1 = src[currentY*W + (W-1)] + 

     M.s8 * (intermediate[currentY*W + (W-1)] - src[currentY*W + (W-1)])

     + M.s9 * (intermediate[currentY*W + (W-2)] - src[currentY*W + (W-1)])

     + M.sa * (intermediate[currentY*W + (W-3)] - src[currentY*W + (W-1)]);

 barrier(CLK_GLOBAL_MEM_FENCE);

   //if current index is W-1
 intermediate[index_equiv] = rightborder_check*temp2Wm1;
  //if current index is W-2
 intermediate[index_equiv] = right_but_one_border_check *
                                           (B * intermediate[index_equiv] +
			 b1 * intermediate[index_equiv + 1] +
		               b2 * temp2W +
			 b3 * temp2Wp1);
   //if current index is W-3
 intermediate[index_equiv] = right_but_two_border_check *
                                           (B * intermediate[index_equiv] +
			 b1 * intermediate[index_equiv + 1] +
		               b2 * intermediate[index_equiv + 2] +
			 b3 * temp2W);
    
barrier(CLK_GLOBAL_MEM_FENCE);

double inner_temp_supplement = leftwardpixels_check *
  B * intermediate[index_equiv] +
  b1 * intermediate[clamp(index_equiv + 1, 0, totalpix)] +
		    b2 * intermediate[clamp(index_equiv + 2, 0, totalpix)] +
		    b3 * intermediate[clamp(index_equiv + 3, 0, totalpix)] ;

 dst[index_equiv] = rightpixels_check * intermediate[index_equiv]  +
		    leftwardpixels_check *  inner_temp_supplement;
  
  if ((currentX == checkintX) && (currentY == checkintY)) {
       printf("initial pass results in %f",  intermediate[index_equiv]);				   
  }

}

