#pragma OPENCL EXTENSION cl_khr_fp64 : enable

float max_func(float divBuffer, float tmp)
{
 float term = 1.f;
 if (tmp > 0.f) term = tmp;
  return fmax(divBuffer / term, 0.f);
  //return fmax(divBuffer / (tmp > 0.f ? tmp : 1.f), 0.f);
} 

__kernel void gauss_3x3_div_whole( __global const float *oldsrc, __global const int *X, __global const int *Y,  const int W, const int H, const double b0, const double b1, const double c0, const double c1, const double c2, __global float *div, __global float *olddst) {

int totalpix = W*H;
// Get the index of the current element to be processed
int index = get_global_id(0);
 index = clamp(index, 0, totalpix);

int currentY = Y[index]; int currentX = X[index];

int index_equiv = currentY * W + currentX;
//printf("index_equiv is %d", index_equiv);

 int checkintX = 100; int checkintY = 100;

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
 int rightborder_check = (1 - toprightcorner_check) * (1 - bottomrightcorner_check) * (1 - clamp((W - 1 - X[index]), 0, 1)); // 1 if the right border, 0 if anything else. If the top right corner or bottom right corner is 1, the result is 0.
 int bottomborder_check = (1 - bottomleftcorner_check) * (1 - bottomrightcorner_check) * (1 - clamp((H - 1 - Y[index]), 0, 1)); // 1 if the bottom border, 0 if anything else. If the bottom left corner or bottom left corner is 1, the result is 0.


   int innerpixels_check = (1 - topleftcorner_check)*(1 - toprightcorner_check)*(1 - bottomleftcorner_check)*(1 - bottomrightcorner_check)*(1 - topborder_check)*(1 - leftborder_check)*(1 - rightborder_check)*(1 - bottomborder_check); 

/**************************************/

   // temp = 2.0;
  
  /*Only one of the right hand terms will be multiplied by 1 rather than 0 - i.e. all of these integer checks but one will be zero.*/

  /*dst[0][0] *= src[0][0];
  dst[0][W - 1] *= src[0][W - 1];
  dst[H - 1][0] *= src[H - 1][0];
  dst[H - 1][W - 1] *= src[H - 1][W - 1];
  
  dst[0][0] *= src[0][0];
  dst[0][W - 1] *= src[0][W - 1];
  dst[H - 1][0] *= src[H - 1][0];
  dst[H - 1][W - 1] *= src[H - 1][W - 1]; */


 float tempa =

    (1 - ((1 - topleftcorner_check) * (1 - toprightcorner_check) * (1 - bottomleftcorner_check) * (1 - bottomrightcorner_check))) * oldsrc[index_equiv]
  + 
    topborder_check * ( b1 * ( oldsrc[clamp(index_equiv - 1, 0, totalpix)] + oldsrc[clamp(index_equiv + 1, 0, totalpix)]  ) + b0 * oldsrc[index_equiv] )
  +
    leftborder_check * ( b1 * ( oldsrc[clamp(index_equiv - W, 0, totalpix)] + oldsrc[clamp(index_equiv + W, 0, totalpix)] ) + b0 * oldsrc[index_equiv] )
  +
   rightborder_check * ( b1 * ( oldsrc[clamp(index_equiv - W, 0, totalpix)] + oldsrc[clamp(index_equiv + W, 0, totalpix)] ) + b0 * oldsrc[index_equiv] )
  +
   bottomborder_check * ( b1 * ( oldsrc[clamp(index_equiv - 1, 0, totalpix)] + oldsrc[clamp(index_equiv + 1, 0, totalpix)] ) + b0 * oldsrc[index_equiv] );

float temp = tempa + 

    innerpixels_check * ( c2 * (oldsrc[clamp(index_equiv - W - 1, 0, totalpix)] + oldsrc[clamp(index_equiv - W + 1, 0, totalpix)] + oldsrc[clamp(index_equiv + W - 1, 0, totalpix)] + oldsrc[clamp(index_equiv + W + 1, 0, totalpix)])
			  + c1 * (oldsrc[clamp(index_equiv - W, 0, totalpix)] + oldsrc[clamp(index_equiv - 1, 0, totalpix)] + oldsrc[clamp(index_equiv + 1, 0, totalpix)] + oldsrc[clamp(index_equiv + W, 0, totalpix)])
					     + c0 * oldsrc[index_equiv]
					     );

/* if (innerpixels_check == 1) {
   temp = innerpixels_check * ( c2 * (oldsrc[clamp(index_equiv - W - 1, 0, totalpix)] + oldsrc[clamp(index_equiv - W + 1, 0, totalpix)] + oldsrc[clamp(index_equiv + W - 1, 0, totalpix)] + oldsrc[clamp(index_equiv + W + 1, 0, totalpix)])
			  + c1 * (oldsrc[clamp(index_equiv - W, 0, totalpix)] + oldsrc[clamp(index_equiv - 1, 0, totalpix)] + oldsrc[clamp(index_equiv + 1, 0, totalpix)] + oldsrc[clamp(index_equiv + W, 0, totalpix)])
					     + c0 * oldsrc[index_equiv]
					     ); } */

  if ((currentX == checkintX) && (currentY == checkintY)) {

    /*printf("DIV Topborder check is %d \n", topborder_check);
    printf("DIV Bottomborder check is %d \n", bottomborder_check);
    printf("DIV Leftborder check is %d \n", leftborder_check);
    printf("DIV Rightborder check is %d \n", rightborder_check);

    printf("DIV Topleftcorner check is %d \n", topleftcorner_check);
    printf("DIV Toprightcorner check is %d \n", toprightcorner_check);
    printf("DIV Bottomleftcorner check is %d \n", bottomleftcorner_check);
    printf("DIV Bottomrightcorner check is %d \n", bottomrightcorner_check); 

    printf("DIV Innerpixels check is %d \n", innerpixels_check);  */


    /*printf("DIV gPu src %d, %d is %f \n", checkintY, checkintX, oldsrc[index_equiv]);
    printf("H is %d, W is %d", H, W);
    printf("1 right border is %d", X[index]);
    printf("2 right border is %d", W - 1 - X[index]);
    printf("3 right border is %d", clamp((W - 1 - X[index]), 0, 1));
    printf("c2 value is %f", c2); */
  }
 
  olddst[index_equiv] = max_func(div[index_equiv], (float)temp);
  
  /*if ((currentX == checkintX) && (currentY == checkintY)) {
    printf("First term is %d\n", (1 - ((1 - topleftcorner_check) * (1 - toprightcorner_check) * (1 - bottomleftcorner_check) * (1 - bottomrightcorner_check))));
    printf("Source is %f\n", oldsrc[index_equiv]);
    printf("DIV gPu src %d, %d intermediate result is %f \n", checkintY, checkintX, temp);
    printf("DIV gPu final result %d, %d is %f \n", checkintY, checkintX, olddst[index_equiv]);
    printf("DivBuffer value %d, %d is %f \n", checkintY, checkintX, div[index_equiv]);
  }
  */
   

}

