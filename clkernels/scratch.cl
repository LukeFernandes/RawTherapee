if ((currentX == checkintX + 1) && (currentY == checkintY)) {
    printf("DIV gPu src 900, 901 intermediate result/temp/tempa is %f,%f,%f \n", c2 * (oldsrc[clamp(index_equiv - W - 1, 0, H*W)] + oldsrc[clamp(index_equiv - W + 1, 0, H*W)] + oldsrc[clamp(index_equiv + W - 1, 0, H*W)] + oldsrc[clamp(index_equiv + W + 1, 0, H*W)])
			  + c1 * (oldsrc[clamp(index_equiv - W, 0, H*W)] + oldsrc[clamp(index_equiv - 1, 0, H*W)] + oldsrc[clamp(index_equiv + 1, 0, H*W)] + oldsrc[clamp(index_equiv + W, 0, H*W)])
	   + c0 * oldsrc[index_equiv], temp, tempa);
    printf("DIV gPu final result 900, 901 is %f \n", olddst[index_equiv]);
    printf("Part 1 DIV 900, 901 is %f \n", c2 * (oldsrc[clamp(index_equiv - W - 1, 0, H*W)] + oldsrc[clamp(index_equiv - W + 1, 0, H*W)] + oldsrc[clamp(index_equiv + W - 1, 0, H*W)] + oldsrc[clamp(index_equiv + W + 1, 0, H*W)]));
    printf("Part 2 DIV 900, 901 is %f \n", c1 * (oldsrc[clamp(index_equiv - W, 0, H*W)] + oldsrc[clamp(index_equiv - 1, 0, H*W)] + oldsrc[clamp(index_equiv + 1, 0, H*W)] + oldsrc[clamp(index_equiv + W, 0, H*W)]));
    printf("Part 2 DIV 900, 901 is %f \n", c0 * oldsrc[index_equiv]);
    printf("DivBuffer value 900, 901 is %f \n", div[index_equiv]);
  }
    if ((currentX == checkintX + 1) && (currentY == checkintY + 1)) {


    printf("Term 1 is %f \n",  (1 - ((1 - topleftcorner_check) * (1 - toprightcorner_check) * (1 - bottomborder_check) * (1 - bottomrightcorner_check))) * oldsrc[index_equiv]);
       printf("Term 2 is %f \n",  topborder_check * ( b1 * ( oldsrc[clamp(index_equiv - 1, 0, H*W)] + oldsrc[clamp(index_equiv + 1, 0, H*W)]  ) + b0 * oldsrc[index_equiv] ));
          printf("Term 3 is %f \n",  leftborder_check * ( b1 * ( oldsrc[clamp(index_equiv - W, 0, H*W)] + oldsrc[clamp(index_equiv + W, 0, H*W)] ) + b0 * oldsrc[index_equiv] ));
	     printf("Term 4 is %f \n",  rightborder_check * ( b1 * ( oldsrc[clamp(W*(currentY - 1) + (W - 1), 0, H*W)] + oldsrc[clamp(W*(currentY + 1) + (W - 1), 0, H*W)] ) + b0 * oldsrc[clamp(W*currentY + (W - 1), 0, H*W)] ));
	        printf("Term 5 is %f \n",   bottomborder_check * ( b1 * ( oldsrc[clamp(W*(H-1) + (currentX - 1), 0, H*W)] + oldsrc[clamp(W*(H-1) + (currentX + 1), 0, H*W)] ) + b0 * oldsrc[clamp(W*(H-1) + currentX, 0, H*W)] ));
		printf("bottom-border term indices are %d,%d,%d \n",  clamp(W*(H-1) + (currentX - 1), 0, H*W), clamp(W*(H-1) + (currentX + 1), 0, H*W), clamp(W*(H-1) + currentX, 0, H*W) );
		printf("bottom-border terms are %f,%f,%f \n", oldsrc[clamp(W*(H-1) + (currentX - 1), 0, H*W)], oldsrc[clamp(W*(H-1) + (currentX + 1), 0, H*W)], oldsrc[clamp(W*(H-1) + currentX, 0, H*W)] );
      
      
    
    printf("DIV gPu src 901, 901 intermediate result/temp/tempa is %f,%f,%f \n",c2 * (oldsrc[clamp(index_equiv - W - 1, 0, H*W)] + oldsrc[clamp(index_equiv - W + 1, 0, H*W)] + oldsrc[clamp(index_equiv + W - 1, 0, H*W)] + oldsrc[clamp(index_equiv + W + 1, 0, H*W)])
	   + c1 * (oldsrc[clamp(index_equiv - W, 0, H*W)] + oldsrc[clamp(index_equiv - 1, 0, H*W)] + oldsrc[clamp(index_equiv + 1, 0, H*W)] + oldsrc[clamp(index_equiv + W, 0, H*W)])
	   + c0 * oldsrc[index_equiv], temp, tempa);
    printf("DIV gPu final result 901, 901 is %f \n", olddst[index_equiv]);
    printf("Part 1 DIV 901, 901 is %f \n", c2 * (oldsrc[clamp(index_equiv - W - 1, 0, H*W)] + oldsrc[clamp(index_equiv - W + 1, 0, H*W)] + oldsrc[clamp(index_equiv + W - 1, 0, H*W)] + oldsrc[clamp(index_equiv + W + 1, 0, H*W)]));
    printf("Part 2 DIV 901, 901 is %f \n", c1 * (oldsrc[clamp(index_equiv - W, 0, H*W)] + oldsrc[clamp(index_equiv - 1, 0, H*W)] + oldsrc[clamp(index_equiv + 1, 0, H*W)] + oldsrc[clamp(index_equiv + W, 0, H*W)]));
    printf("Part 2 DIV 901, 901 is %f \n", c0 * oldsrc[index_equiv]);
     printf("DivBuffer value 901, 901 is %f \n", div[index_equiv]);
  }
		  
