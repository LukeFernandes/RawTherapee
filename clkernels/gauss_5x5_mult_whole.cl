#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void gauss_5x5_mult_whole( __global const float *oldsrc, __global const int *X, __global const int *Y,  const int W, const int H, const float c21, const float c20, const float c11, const float c10, const float c00, __global float *olddst) {
  
int totalpix = (W*H)-1;;
// Get the index of the current element to be processed
int index = get_global_id(0);
 index = clamp(index, 0, totalpix);

int currentY = Y[index]; int currentX = X[index];

int index_equiv = currentY * W + currentX;
//printf("index_equiv is %d", index_equiv);

 int checkintX = 100; int checkintY = 100;

/*This section avoids conditioal branching*/

 // printf("borders done");
 
// Check if on the borders
 int topborder_check = (1 - clamp(Y[index], 0, 1)); // 1 if the top border, 0 if anything else. If the top left corner or top right corner is 1, the result is 0.
 int top_but_one_border_check = (1 - clamp(Y[index] - 1, 0, 1)); // 1 if the top border but one, 0 if anything else. If the top left corner or top right corner is 1, the result is 0.
 int leftborder_check = (1 - clamp(X[index], 0, 1)); // 1 if the left border, 0 if anything else. If the top left corner or bottom left corner is 1, the result is 0.
 int left_but_one_border_check = (1 - clamp(X[index] - 1, 0, 1)); // 1 if the left border but one, 0 if anything else. If the top left corner or bottom left corner is 1, the result is 0.
 int rightborder_check = (1 - clamp((W - 1 - X[index]), 0, 1)); // 1 if the right border, 0 if anything else. If the top right corner or bottom right corner is 1, the result is 0.
 int right_but_one_border_check = (1 - clamp(( W - 1 - (X[index] + 1) ), 0, 1)); // 1 if the right border but one, 0 if anything else. If the top right corner or bottom right corner is 1, the result is 0.
 int bottomborder_check = (1 - clamp((H - 1 - Y[index]), 0, 1)); // 1 if the bottom border, 0 if anything else. If the bottom left corner or bottom left corner is 1, the result is 0.
 int bottom_but_one_border_check = (1 - clamp(( H - 1 - (Y[index] + 1) ), 0, 1)); // 1 if the bottom border but one, 0 if anything else. If the bottom left corner or bottom left corner is 1, the result is 0.


   int innerpixels_check = (1 - top_but_one_border_check)*(1 - left_but_one_border_check)*(1 - right_but_one_border_check)*(1 - bottom_but_one_border_check)*(1 - topborder_check)*(1 - leftborder_check)*(1 - rightborder_check)*(1 - bottomborder_check); 

float temp_pre_mult = ( c21 * ( oldsrc[clamp(index_equiv - (2 * W) - 1, 0, totalpix)]
				     + oldsrc[clamp(index_equiv - (2 * W) + 1, 0, totalpix)]
				     + oldsrc[clamp(index_equiv - W - 2, 0, totalpix)]
				     + oldsrc[clamp(index_equiv - W + 2, 0, totalpix)]
				     + oldsrc[clamp(index_equiv + W - 2, 0, totalpix)]
				     + oldsrc[clamp(index_equiv + W + 2, 0, totalpix)]
				     + oldsrc[clamp(index_equiv + (2 * W) - 1, 0, totalpix)]
				     + oldsrc[clamp(index_equiv + (2 * W) + 1, 0, totalpix)] )
				 
			    + c20 * (  oldsrc[clamp(index_equiv - (2 * W), 0, totalpix)]
				       + oldsrc[clamp(index_equiv - 2, 0, totalpix)]
				       + oldsrc[clamp(index_equiv + 2, 0, totalpix)]
				       + oldsrc[clamp(index_equiv + (2 * W), 0, totalpix)] )
				 
			    + c11 * (  oldsrc[clamp(index_equiv - W - 1, 0, totalpix)]
				       + oldsrc[clamp(index_equiv - W + 1, 0, totalpix)]
				       + oldsrc[clamp(index_equiv + W - 1, 0, totalpix)]
				       + oldsrc[clamp(index_equiv + W + 1, 0, totalpix)] )

			    + c10 * (  oldsrc[clamp(index_equiv - W, 0, totalpix)]
				       + oldsrc[clamp(index_equiv - 1, 0, totalpix)]
				       + oldsrc[clamp(index_equiv + 1, 0, totalpix)]
				       + oldsrc[clamp(index_equiv + W, 0, totalpix)] ) 

			    + c00  *  oldsrc[index_equiv]
					     );

 float mult = 1.f*(1 - innerpixels_check) +
   temp_pre_mult*innerpixels_check;
  olddst[index_equiv] = olddst[index_equiv] * mult;

  
  /*if ((currentX == checkintX) && (currentY == checkintY)) {
    printf("Source is %f\n", oldsrc[index_equiv]);
    printf("MULT 5x5 gPu src %d, %d  intermbgaediate is %f \n", checkintY, checkintX, temp_pre_mult);
    printf("MULT 5x5 gPu final result %d, %d is %f \n", checkintY, checkintX, olddst[index_equiv]);
   
    }*/
  
}

