#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void gauss_3x3_standard_whole( __global const float *oldsrc, __global const int *X, __global const int *Y,  const int W, const int H, const double b0, const double b1, const double c0, const double c1, const double c2, __global float *olddst) {
 
// Get the index of the current element to be processed
int index = get_global_id(0);

int currentY = Y[index]; int currentX = X[index];

int index_equiv = currentY * W + currentX;
//printf("index_equiv is %d", index_equiv);

 int checkintX = 900; int checkintY = 900;

/*This section avoids conditioal branching*/

// Check to make sure not top left or right corners at first
 int topleftcorner_check = 1 - ( clamp(X[index], 0, 1) | clamp(Y[index], 0, 1) ); //1 if both X and Y are 0. 0 if either X or Y are 0. Using NOR operator (1 - |).

 int toprightcorner_check = 1 - ( clamp((W - 1 - X[index]), 0, 1) | clamp(Y[index], 0, 1) ); //1 if both X is (W-1) and Y is 0. 0 otherwise. Using NOR operator (1 - |).

// Check to make sure not bottom left or right corners at first
 int bottomleftcorner_check = 1 - ( clamp(X[index], 0, 1) |  clamp((H - 1 - Y[index]), 0, 1) ); //1 if both X and Y are 0. 0 if either X or Y are 0. Using NOR operator (1 - |).
 int bottomrightcorner_check = 1 - ( clamp((W - 1 - X[index]), 0, 1) | clamp((H - 1 - Y[index]), 0, 1) ); //1 if both X is (W-1) and Y is 0. 0 otherwise. Using NOR operator (1 - |).

 // printf("borders done");
 
// Check if on the borders
 int topborder_check = (1 - topleftcorner_check) * (1 - toprightcorner_check) * (1 - clamp(Y[index], 0, 1)); // 1 if the top border, 0 if anything else. If the top left corner or top right corner is 1, the result is 0.
 int leftborder_check = (1 - topleftcorner_check) * (1 - bottomleftcorner_check) * (1 - clamp(X[index], 0, 1)); // 1 if the left border, 0 if anything else. If the top left corner or bottom left corner is 1, the result is 0.
 int rightborder_check = (1 - toprightcorner_check) * (1 - bottomrightcorner_check) * (1 - clamp((W - X[index]), 0, 1)); // 1 if the right border, 0 if anything else. If the top right corner or bottom right corner is 1, the result is 0.
 int bottomborder_check = (1 - bottomleftcorner_check) * (1 - bottomrightcorner_check) * (1 - clamp((H - Y[index]), 0, 1)); // 1 if the bottom border, 0 if anything else. If the bottom left corner or bottom left corner is 1, the result is 0.

   int innerpixels_check = (1 - topleftcorner_check)*(1 - toprightcorner_check)*(1 - bottomleftcorner_check)*(1 - bottomrightcorner_check)*(1 - topborder_check)*(1 - leftborder_check)*(1 - rightborder_check)*(1 - bottomborder_check); 

/**************************************/

   double temp = 2.0;
  
  /*Only one of the right hand terms will be multiplied by 1 rather than 0 - i.e. all of these integer checks but one will be zero.*/

  /*dst[0][0] *= src[0][0];
  dst[0][W - 1] *= src[0][W - 1];
  dst[H - 1][0] *= src[H - 1][0];
  dst[H - 1][W - 1] *= src[H - 1][W - 1];
  
  dst[0][0] *= src[0][0];
  dst[0][W - 1] *= src[0][W - 1];
  dst[H - 1][0] *= src[H - 1][0];
  dst[H - 1][W - 1] *= src[H - 1][W - 1]; */


  temp =

    (1 - ((1 - topleftcorner_check) * (1 - toprightcorner_check) * (1 - bottomborder_check) * (1 - bottomrightcorner_check))) * oldsrc[index_equiv]
  + 
    topborder_check * ( b1 * ( oldsrc[clamp(index_equiv - 1, 0, H*W)] + oldsrc[clamp(index_equiv + 1, 0, H*W)]  ) + b0 * oldsrc[index_equiv] )
  +
    leftborder_check * ( b1 * ( oldsrc[clamp(index_equiv - W, 0, H*W)] + oldsrc[clamp(index_equiv + W, 0, H*W)] ) + b0 * oldsrc[index_equiv] )
  +
    rightborder_check * ( b1 * ( oldsrc[clamp(W*(currentY - 1) + (W - 1), 0, H*W)] + oldsrc[clamp(W*(currentY + 1) + (W - 1), 0, H*W)] ) + b0 * oldsrc[clamp(W*currentY + (W - 1), 0, H*W)] )
  +
    bottomborder_check * ( b1 * ( oldsrc[clamp(W*(H-1) + (currentX - 1), 0, H*W)] + oldsrc[clamp(W*(H-1) + (currentX + 1), 0, H*W)] ) + b0 * oldsrc[clamp(W*(H-1) + currentX, 0, H*W)] )

  +

    innerpixels_check * ( c2 * (oldsrc[clamp(index_equiv - W - 1, 0, H*W)] + oldsrc[clamp(index_equiv - W + 1, 0, H*W)] + oldsrc[clamp(index_equiv + W - 1, 0, H*W)] + oldsrc[clamp(index_equiv + W + 1, 0, H*W)])
			  + c1 * (oldsrc[clamp(index_equiv - W, 0, H*W)] + oldsrc[clamp(index_equiv - 1, 0, H*W)] + oldsrc[clamp(index_equiv + 1, 0, H*W)] + oldsrc[clamp(index_equiv + W, 0, H*W)])
					     + c0 * oldsrc[index_equiv]
					     );

  if ((currentX == checkintX) && (currentY == checkintY)) {
    /* printf("STAND Topborder check is %d \n", topborder_check);
    printf("STAND Bottomborder check is %d \n", bottomborder_check);
    printf("STAND Leftborder check is %d \n", leftborder_check);
    printf("STAND Rightborder check is %d \n", rightborder_check);

    printf("STAND Topleftcorner check is %d \n", topleftcorner_check);
    printf("STAND Toprightcorner check is %d \n", toprightcorner_check);
    printf("STAND Bottomleftcorner check is %d \n", bottomleftcorner_check);
    printf("STAND Bottomrightcorner check is %d \n", bottomrightcorner_check);

    printf("STAND Innerpixels check is %d \n", innerpixels_check); */

     printf("STAND gPu src 900, 900 is %f \n", oldsrc[index_equiv]);;
  }
 
  olddst[index_equiv] = temp;
  
  if ((currentX == checkintX) && (currentY == checkintY)) printf("gPu STAND final result is %f \n", olddst[index_equiv]);
		  

}
