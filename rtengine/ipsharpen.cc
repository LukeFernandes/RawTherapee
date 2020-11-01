/*
 *  This file is part of RawTherapee.
 *
 *  Copyright (c) 2004-2010 Gabor Horvath <hgabor@rawtherapee.com>
 *
 *  RawTherapee is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  RawTherapee is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with RawTherapee.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "bilateral2.h"
#include "cieimage.h"
#include "gauss.h"
#include "improcfun.h"
#include "jaggedarray.h"
#include "labimage.h"
#include "opthelper.h"
#include "procparams.h"
#include "rt_algo.h"
#include "rt_math.h"
#include "settings.h"
#include "sleef.h"
#include "math.h"
#include "assert.h"
#include <map>
#include <chrono>

//#define BENCHMARK
#include "StopWatch.h"
#include "rt_algo.h"
#include <CL/cl.h>
#include "stdio.h"
#include "OpenCL_support.h"
#define MAX_SOURCE_SIZE (0x100000)

using namespace std;

namespace {

void sharpenHaloCtrl (float** luminance, float** blurmap, float** base, float** blend, int W, int H, const procparams::SharpeningParams &sharpenParam)
{

    const float scale = (100.f - sharpenParam.halocontrol_amount) * 0.01f;
    const float sharpFac = sharpenParam.amount * 0.01f;
    float** nL = base;

#ifdef _OPENMP
    #pragma omp parallel for
#endif

    for (int i = 2; i < H - 2; i++) {
        float max1 = 0, max2 = 0, min1 = 0, min2 = 0;

        for (int j = 2; j < W - 2; j++) {
            // compute 3 iterations, only forward
            float np1 = 2.f * (nL[i - 2][j] + nL[i - 2][j + 1] + nL[i - 2][j + 2] + nL[i - 1][j] + nL[i - 1][j + 1] + nL[i - 1][j + 2] + nL[i]  [j] + nL[i]  [j + 1] + nL[i]  [j + 2]) / 27.f + nL[i - 1][j + 1] / 3.f;
            float np2 = 2.f * (nL[i - 1][j] + nL[i - 1][j + 1] + nL[i - 1][j + 2] + nL[i]  [j] + nL[i]  [j + 1] + nL[i]  [j + 2] + nL[i + 1][j] + nL[i + 1][j + 1] + nL[i + 1][j + 2]) / 27.f + nL[i]  [j + 1] / 3.f;
            float np3 = 2.f * (nL[i]  [j] + nL[i]  [j + 1] + nL[i]  [j + 2] + nL[i + 1][j] + nL[i + 1][j + 1] + nL[i + 1][j + 2] + nL[i + 2][j] + nL[i + 2][j + 1] + nL[i + 2][j + 2]) / 27.f + nL[i + 1][j + 1] / 3.f;

            // Max/Min of all these deltas and the last two max/min
            float maxn = rtengine::max(np1, np2, np3);
            float minn = rtengine::min(np1, np2, np3);
            float max_ = rtengine::max(max1, max2, maxn);
            float min_ = rtengine::min(min1, min2, minn);

            // Shift the queue
            max1 = max2;
            max2 = maxn;
            min1 = min2;
            min2 = minn;
            float labL = luminance[i][j];

            if (max_ < labL) {
                max_ = labL;
            }

            if (min_ > labL) {
                min_ = labL;
            }

            // deviation from the environment as measurement
            float diff = nL[i][j] - blurmap[i][j];

            constexpr float upperBound = 2000.f;  // WARNING: Duplicated value, it's baaaaaad !
            float delta = sharpenParam.threshold.multiply<float, float, float>(
                              rtengine::min(fabsf(diff), upperBound),   // X axis value = absolute value of the difference
                              sharpFac * diff               // Y axis max value = sharpening.amount * signed difference
                          );
            float newL = labL + delta;

            // applying halo control
            if (newL > max_) {
                newL = max_ + (newL - max_) * scale;
            } else if (newL < min_) {
                newL = min_ - (min_ - newL) * scale;
            }

            luminance[i][j] = intp(blend[i][j], newL, luminance[i][j]);
        }
    }
}

void dcdamping (float** aI, float** aO, float damping, int W, int H)
{

    const float dampingFac = -2.f / (damping * damping);

#ifdef __SSE2__
    vfloat Iv, Ov, Uv, zerov, onev, fourv, fivev, dampingFacv, Tv, Wv, Lv;
    zerov = _mm_setzero_ps();
    onev = F2V(1.f);
    fourv = F2V(4.f);
    fivev = F2V(5.f);
    dampingFacv = F2V(dampingFac);
#endif
#ifdef _OPENMP
    #pragma omp for
#endif

    for (int i = 0; i < H; i++) {
        int j = 0;
#ifdef __SSE2__

        for (; j < W - 3; j += 4) {
            Iv = LVFU(aI[i][j]);
            Ov = LVFU(aO[i][j]);
            Lv = xlogf(Iv / Ov);
            Wv = Ov - Iv;
            Uv = (Ov * Lv + Wv) * dampingFacv;
            Uv = vminf(Uv, onev);
            Tv = Uv * Uv;
            Tv = Tv * Tv;
            Uv = Tv * (fivev - Uv * fourv);
            Uv = (Wv / Iv) * Uv + onev;
            Uv = vselfzero(vmaskf_gt(Iv, zerov), Uv);
            Uv = vselfzero(vmaskf_gt(Ov, zerov), Uv);
            STVFU(aI[i][j], Uv);
        }

#endif

        for(; j < W; j++) {
            float I = aI[i][j];
            float O = aO[i][j];

            if (O <= 0.f || I <= 0.f) {
                aI[i][j] = 0.f;
                continue;
            }

            float U = (O * xlogf(I / O) - I + O) * dampingFac;
            U = rtengine::min(U, 1.0f);
            U = U * U * U * U * (5.f - U * 4.f);
            aI[i][j] = (O - I) / I * U + 1.f;
        }
    }
}

void OpenCL_intial_setup_and_max(float* lum, JaggedArray<float> &tmpA, OpenCL_helper* helper, int W, int H, cl_mem input_mem_obj, cl_mem ret_mem_obj,  size_t _global_item_size, float* _read_storage)
{

      cl_kernel mykernel;
      kernel_tag maxtag = maxkernel;
      //this kernel is for maximising luminance
      mykernel = helper->reuse_or_create_kernel("maxkernel", "opencl_max.cl", "opencl_max");         
 
      cl_int error_code = 0;
      //create the OpenCL memory objects that coresspond to the input luminance and the return value
	
    error_code = clEnqueueWriteBuffer(helper->command_queue, input_mem_obj, CL_TRUE, 0, W*H*sizeof(float), lum, 0, NULL, NULL);
    if (error_code != 0) printf("OpenCL Error code (0 is success):%d\n", error_code); 
      
      error_code = clSetKernelArg(mykernel, 0, sizeof(cl_mem), (void *)&input_mem_obj);
    if (error_code != 0) printf("OpenCL Error code (0 is success):%d\n", error_code);  
      error_code = clSetKernelArg(mykernel, 1, sizeof(cl_mem), (void *)&ret_mem_obj);
    if (error_code != 0) printf("OpenCL Error code (0 is success):%d\n", error_code); 
      size_t global_item_size = H*W; 
    
      error_code = clEnqueueNDRangeKernel(helper->command_queue, mykernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
      
    if (error_code != 0) printf("OpenCL Error code (0 is success):%d\n", error_code);
 
       error_code = clEnqueueReadBuffer(helper->command_queue, ret_mem_obj, CL_TRUE, 0, W*H*sizeof(float), _read_storage, 0, NULL, NULL);
    if (error_code != 0) printf("OpenCL Error code (0 is success):%d\n", error_code);
 
       //turn 2D array into jagged array

       /* for (int i = 0; i < H; i++)
	   {
	    for (int j = 0; j < W; j++)
	       {
		 tmpA[i][j] = _read_storage[i*W + j];
	       }
	       } */

}

void OpenCL_intp_1 (OpenCL_helper* helper, int W, int H, cl_mem input_mem_obj, cl_mem blur_mem_obj, cl_mem blend_mem_obj, float* _read_storage) {

                 cl_int error_code = 0;
	   size_t global_item_size = H*W;
                 cl_kernel intp_kernel = helper->reuse_or_create_kernel("intp1", "intp_plus.cl", "intp_plus");

	    error_code = clSetKernelArg(intp_kernel, 0, sizeof(cl_mem), (void *)&blend_mem_obj);
	    error_code = clSetKernelArg(intp_kernel, 1, sizeof(cl_mem), (void *)&input_mem_obj);
	    error_code = clSetKernelArg(intp_kernel, 2, sizeof(cl_mem), (void *)&blur_mem_obj);

	    error_code = clEnqueueNDRangeKernel(helper->command_queue, intp_kernel, 1, nullptr, &global_item_size, nullptr, 0, nullptr, nullptr); 
	    error_code = clEnqueueReadBuffer(helper->command_queue, blur_mem_obj, CL_TRUE, 0, W*H*sizeof(float), _read_storage, 0, nullptr, nullptr); 	       
		      
}

void OpenCL_intp_2 (OpenCL_helper* helper, int W, int H, cl_mem blend_mem_obj, cl_mem tmpI_mem_obj, cl_mem lum_mem_obj, const float* amount) {

                cl_int error_code = 0;
	  size_t global_item_size = H*W; 
                cl_kernel intp_kernel = helper->reuse_or_create_kernel("intp2", "intp_plus_version2.cl", "intp_plus_version2");

	   error_code = clSetKernelArg(intp_kernel, 0, sizeof(cl_mem), (void *)&blend_mem_obj);
	   error_code = clSetKernelArg(intp_kernel, 1, sizeof(cl_mem), (void *)&tmpI_mem_obj);
	   error_code = clSetKernelArg(intp_kernel, 2, sizeof(cl_mem), (void *)&lum_mem_obj);
	   error_code = clSetKernelArg(intp_kernel, 3, sizeof(cl_float), (void *)amount);

	   error_code = clEnqueueNDRangeKernel(helper->command_queue, intp_kernel, 1, nullptr, &global_item_size, nullptr, 0, nullptr, nullptr);		      
}

void OpenCL_intp_3 (OpenCL_helper* helper, int W, int H, cl_mem blend_mem_obj, cl_mem lum_mem_obj, cl_mem blur_mem_obj) {

                 cl_int error_code = 0;
	   size_t global_item_size = H*W; 
                 cl_kernel intp_kernel = helper->reuse_or_create_kernel("intp3", "intp_plus_version3.cl", "intp_plus_version3");

	   error_code = clSetKernelArg(intp_kernel, 0, sizeof(cl_mem), (void *)&blend_mem_obj);
	   error_code = clSetKernelArg(intp_kernel, 1, sizeof(cl_mem), (void *)&lum_mem_obj);
	   error_code = clSetKernelArg(intp_kernel, 2, sizeof(cl_mem), (void *)&blur_mem_obj);

	   error_code = clEnqueueNDRangeKernel(helper->command_queue, intp_kernel, 1, nullptr, &global_item_size, nullptr, 0, nullptr, nullptr); 	      
}



}

namespace rtengine
{

  void ImProcFunctions::deconvsharpening2 (float** luminance, float** tmp, const float * const * blend, int W, int H, const procparams::SharpeningParams &sharpenParam, double Scale)
{
    if (sharpenParam.deconvamount == 0 && sharpenParam.blurradius < 0.25) {
        return;
    }
BENCHFUN
    JaggedArray<float> tmpI(W, H);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++) {
            tmpI[i][j] = max(luminance[i][j], 0.f);
        }
    }

    JaggedArray<float>* blurbuffer = nullptr;

    if (sharpenParam.blurradius >= 0.25) {
        blurbuffer = new JaggedArray<float>(W, H);
        JaggedArray<float> &blur = *blurbuffer;
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            gaussianBlur(tmpI, blur, W, H, sharpenParam.blurradius);
#ifdef _OPENMP
            #pragma omp for
#endif
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    blur[i][j] = intp(blend[i][j], luminance[i][j], std::max(blur[i][j], 0.0f));
                }
            }
        }
    }
    const float damping = sharpenParam.deconvdamping / 5.0;
    const bool needdamp = sharpenParam.deconvdamping > 0;
    const double sigma = sharpenParam.deconvradius / Scale;
    const float amount = sharpenParam.deconvamount / 100.f;

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        for (int k = 0; k < sharpenParam.deconviter; k++) {
            if (!needdamp) {
                // apply gaussian blur and divide luminance by result of gaussian blur
	      gaussianBlur(tmpI, tmp, W, H, sigma, false, nullptr, GAUSS_DIV, luminance);
            } else {
                // apply gaussian blur + damping
                gaussianBlur(tmpI, tmp, W, H, sigma);
                dcdamping(tmp, luminance, damping, W, H);
            }
            gaussianBlur(tmp, tmpI, W, H, sigma, false, nullptr, GAUSS_MULT);
        } // end for

#ifdef _OPENMP
        #pragma omp for
#endif

        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                luminance[i][j] = intp(blend[i][j] * amount, max(tmpI[i][j], 0.0f), luminance[i][j]);
            }
        }

        if (sharpenParam.blurradius >= 0.25) {
            JaggedArray<float> &blur = *blurbuffer;
#ifdef _OPENMP
        #pragma omp for
#endif
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    luminance[i][j] = intp(blend[i][j], luminance[i][j], max(blur[i][j], 0.0f));
                }
            }
        }
    } // end parallel
    delete blurbuffer;
}

void ImProcFunctions::deconvsharpening (float** luminance, float** tmp, const float * const * blend, int W, int H, const procparams::SharpeningParams &sharpenParam, double Scale)
{
  // Record start time
  auto start_ch= std::chrono::high_resolution_clock::now();
  
  printf("\n\nDeconvolution sharpening\n\nW is %d and H is %d\n\n", W, H);
  OpenCL_use gpuSelected;
  /*procMethod records the drop down selection. This drop down will need to be removed in production code, but it is useful for debugging/comparison
  GPU - OpenCL is used, no printing for speed comparison
  CPU - CPU is used, no printing for speed comparison
  Debug - both OpenCL and CPU used interleaved, error and debug information printed
  */
  if (sharpenParam.procMethod == "GPU")
    gpuSelected = true_;
  else if (sharpenParam.procMethod == "CPU") gpuSelected = false_;
  else if (sharpenParam.procMethod == "Debug") gpuSelected = debug_;
    if (sharpenParam.deconvamount == 0 && sharpenParam.blurradius < 0.25f) {
        return;
    }
BENCHFUN
  JaggedArray<float> tmpI(W, H); 

     int sampleJ = 100;
     int sampleI = 100;

     //first CPU - max operation
    if (OpenCL_helper::OpenCL_usable(gpuSelected) == false_ || debug_)
      {
#ifdef _OPENMP
    #pragma omp parallel for
#endif
	for (int i = 0; i < H; i++)
	  {
	    for(int j = 0; j < W; j++) {
	      tmpI[i][j] = max(luminance[i][j], 0.f); 
	    }
	  }
      }
    /*OpenCL helper class setup ***************** */
            OpenCL_helper* helper;
    //set up OpenCL if not already set up
    if (OpenCL_helper::OpenCL_usable(gpuSelected) == true_ || debug_) {
      if (this->helper == nullptr) {
          helper = new OpenCL_helper();
          this->helper = helper;
      }
      else {
	   helper = this->helper;     
      }
     }

    //Timing how long it took to setup
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start_ch;
    std::cout << "OpenCL setup time: " << elapsed.count() << " s\n";
    
    /*OpenCL definitions ***************** */
    float* lum;
    cl_kernel max_kernel; // provides a handle to the fmax kernel we are going to create on GPU
    cl_mem lum_mem_obj, tmpI_mem_obj, blend_mem_obj, blur_mem_obj, tmp_mem_obj, index_X_mem_obj, index_Y_mem_obj = nullptr; //Produces GPU equivalents in memory - just empty handles at the moment - will be initialised later

    int *Xindex, *Yindex;  float *read_storage; 
    size_t global_item_size = H*W; size_t local_item_size = 64;
    cl_int error_code = 1;  

    float *blend1d;  float *blur1d;

    /*OpenCL section 1 - max ***************** */  
    if (OpenCL_helper::OpenCL_usable(gpuSelected) == true_ || OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
      {       
	//turn array of arrays into 1d array
	lum = new float[W*H]();
	OpenCL_helper::ArrayofArrays_to_1d_array(lum, luminance, W, H);

	max_kernel = helper->reuse_or_create_kernel("maxkernel", "opencl_max.cl", "opencl_max"); //this kernel is just using the fmax intrinsic        

	lum_mem_obj = helper->reuse_or_create_buffer("luminance", W, H, CL_MEM_READ_ONLY);
	tmpI_mem_obj = helper->reuse_or_create_buffer("tmpI", W, H, CL_MEM_READ_WRITE);
      
	read_storage = new float[W*H]();

	//perform the fmax operation on GPU
	OpenCL_intial_setup_and_max(lum, tmpI, helper, W, H, lum_mem_obj, tmpI_mem_obj, global_item_size, read_storage);
	delete[] lum;
      
      }
	  
    // calculate contrast based blend factors to reduce sharpening in regions with low contrast
    JaggedArray<float> blend_jagged_array(W, H);
    float contrast = sharpenParam.contrast / 100.f;
    buildBlendMask(luminance, blend_jagged_array, W, H, contrast, 1.f);

    JaggedArray<float>* blurbuffer = nullptr;

    //Prepare blend GPU buffer - need to do this outside the conditional >0.25 below
    if (OpenCL_helper::OpenCL_usable(gpuSelected) == true_ || OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
    {
     blend1d = new float[W*H]();
     helper->JaggedArray_to_1d_array(blend1d, &blend_jagged_array, W, H);
     blend_mem_obj = helper->reuse_or_create_buffer("blend", W, H, CL_MEM_READ_WRITE, blend1d);
     delete[] blend1d;
    }
    
    if (sharpenParam.blurradius >= 0.25) {

        blurbuffer = new JaggedArray<float>(W, H);
        JaggedArray<float> &blur = *blurbuffer;

	/**** OpenCL section 2 ***************** 
	  Transposition of 'gaussianBlur(tmpI, blur, W, H, sharpenParam.blurradius);' into OpenCL
	*************************************/
	
	if (OpenCL_helper::OpenCL_usable(gpuSelected) == true_ || OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
	  {
                  blur1d = new float[W*H]();
	    helper->JaggedArray_to_1d_array(blur1d, &blur, W, H);
	    blur_mem_obj = helper->reuse_or_create_buffer("blur", W, H, CL_MEM_READ_WRITE, blur1d);
	    delete[] blur1d;

	    //perform the gaussian blur, 1 iteration
	    OpenCLgaussianBlur(tmpI, blur, gpuSelected, helper, 1, tmpI_mem_obj, blur_mem_obj, W, H, sharpenParam.blurradius, nullptr, false);
	  }
	
	/**** CPU section 2 *****************/
	if (OpenCL_helper::OpenCL_usable(gpuSelected) == false_ || OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
	  {
#ifdef _OPENMP
#pragma omp parallel
#endif
	    {
	      gaussianBlur(tmpI, blur, W, H, sharpenParam.blurradius);
	    }
	  }

	//if using debug mode - check results        
        if (OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
	  {
	   //sample and compare blur over a range
	    error_code = clEnqueueReadBuffer(helper->command_queue, blur_mem_obj, CL_TRUE, 0, W*H*sizeof(float), read_storage, 0, nullptr, nullptr);
	    printf("\n\nSample and compare what is in the blur buffer over the 197-203 square after the first gaussian blur\n\n**********************\n");
	    for (int i = 197; i < 203; i++)
	      {
	        for (int j = 197; j < 203; j++)
		  {
		   printf("\n   OpenCl GPU Difference: CPU is %f, gPu is %f at %d,%d", blur[i][j], read_storage[i*W + j], i, j);  
		  }
	      }
	    printf("\n\n**********************");
	  }
	
	    /**** OpenCL section 3 ***************** 
		  Transposition of 'blur[i][j] = intp(blend_jagged_array[i][j], luminance[i][j], std::max(blur[i][j], 0.0f));' into OpenCL
	    *************************************/
	 if (OpenCL_helper::OpenCL_usable(gpuSelected) == true_ || OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
	   {
	    assert(helper->buffer_set.find("luminance") != helper->buffer_set.end());
	    OpenCL_intp_1(helper, W, H, lum_mem_obj, blur_mem_obj, blend_mem_obj, read_storage);		      
	  }
	
	    /**** CPU section 3 *****************/
	if (OpenCL_helper::OpenCL_usable(gpuSelected) == false_ || OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
	  {
           #ifdef _OPENMP
                 #pragma omp for
           #endif

	   for (int i = 0; i < H; ++i)
	     {
                  for (int j = 0; j < W; ++j)
		    {		  
		  blur[i][j] = intp(blend_jagged_array[i][j], luminance[i][j], blur[i][j]);//std::max(blur[i][j], 0.0f));	    
                               }
                   }	   
	  }

	if (OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
	  {
	    //sample and compare blur over a range
	    error_code = clEnqueueReadBuffer(helper->command_queue, blur_mem_obj, CL_TRUE, 0, W*H*sizeof(float), read_storage, 0, nullptr, nullptr);
	    printf("\n\nSample and compare what is in the blur buffer over the 197-203 square after the intp operation\n\n**********************\n");
	    for (int i = 197; i < 203; i++)
	      {
	        for (int j = 197; j < 203; j++)
		  {
		      printf("\n   intp Difference: CPU is %f, CPU blend is %f, GPU is %f at %d,%d", blur[i][j], blend_jagged_array[i][j], read_storage[i*W + j], i, j); 
		  }
	      }
	    printf("\n\n**********************");
	  }

	  
    }
    const float damping = sharpenParam.deconvdamping / 5.0;
    const bool needdamp = sharpenParam.deconvdamping > 0;
    const double sigma = sharpenParam.deconvradius / Scale;
    const float amount = sharpenParam.deconvamount / 100.f;
    
    //if using gPu, need to write tmp into a gPu buffer
    if (OpenCL_helper::OpenCL_usable(gpuSelected) == true_ || OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
      {
	tmp_mem_obj = helper->reuse_or_create_buffer("tmp", W, H, CL_MEM_READ_WRITE);
    
	float* tmp1d = new float[W*H]();
	OpenCL_helper::ArrayofArrays_to_1d_array(tmp1d, tmp, W, H);
	error_code = clEnqueueWriteBuffer(helper->command_queue, tmp_mem_obj, CL_TRUE, 0, W*H*sizeof(float), tmp1d, 0, NULL, NULL);
	delete[] tmp1d;
      }

    /* in debug mode, we check that before going into multi-iteration blur, the values from the CPU and gPu line up */
     if (OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
      {
	printf("\n\nCheck: luminance CPU is %f\n", luminance[sampleI][sampleJ]); printf("Check: luminance gPu is %f\n", helper->debug_get_value_from_GPU_buffer(lum_mem_obj, sampleI, sampleJ, W, H));
	printf("Check: gPu BUFFER tmpI (src) for %d,%d is %f \n", sampleI, sampleJ, helper->debug_get_value_from_GPU_buffer(tmpI_mem_obj, sampleI, sampleJ, W, H)); printf("Check: CPU tmpI (src) is %d,%d is %f \n", sampleI, sampleJ, tmpI[sampleI][sampleJ]);
	printf("Check: gPu BUFFER blend for %d,%d is %f \n", sampleI, sampleJ, helper->debug_get_value_from_GPU_buffer(blend_mem_obj, sampleI, sampleJ, W, H)); printf("Check: CPU blend for %d,%d is %f \n", sampleI, sampleJ, blend_jagged_array[sampleI][sampleJ]);
    
	if (blur_mem_obj != nullptr) printf("Check: gPu BUFFER Blur for %d,%d is %f \n", sampleI, sampleJ, helper->debug_get_value_from_GPU_buffer(blur_mem_obj, sampleI, sampleJ, W, H));
	if (blurbuffer != nullptr) printf("Check: CPU Blur for %d,%d is %f \n\n\n", sampleI, sampleJ, (*blurbuffer)[sampleI][sampleJ]);
      }

    /**** CPU section 4 - multi iteration blur *****************/
   if (OpenCL_helper::OpenCL_usable(gpuSelected) == false_ || OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
     {
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start_ch;
    printf("**********CPU multi-iteration blur**********\n\n");
    std::cout << "   Time before CPU multiple gauss: " << elapsed.count() << " s\n\n";
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
       for (int k = 0; k < sharpenParam.deconviter; k++) {
	   if (!needdamp) {
	     // apply gaussian blur and divide luminance by result of gaussian blur
	     gaussianBlur(tmpI, tmp, W, H, sigma, false, nullptr, GAUSS_DIV, luminance); 
	   } else {
	   // apply gaussian blur + damping
	     gaussianBlur(tmpI, tmp, W, H, sigma);
	     dcdamping(tmp, luminance, damping, W, H);
	 }
	 gaussianBlur(tmp, tmpI, W, H, sigma, false, nullptr, GAUSS_MULT);
	 #pragma omp master
	 {
	 if (OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
	   printf("   CPU iteration no. %d: Post mult  tmpI/src %d,%d is %f \n", k, sampleI, sampleJ, tmpI[sampleI][sampleJ]);
	 }
       } // end for
    }
       /*****************************/
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start_ch;
    std::cout << "\n   Time after CPU multiple gauss: " << elapsed.count() << " s\n********************\n";
       fflush(stdout);
     }
     /***************************/

    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start_ch;
    std::cout << "Time before gPu multiple gauss: " << elapsed.count() << " s\n";
   
     /**** OpenCL section 4 ***************** 
      Transposition of above loop into OpenCL
     ****************************************/
    if (OpenCL_helper::OpenCL_usable(gpuSelected) == true_ || OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
      {
	/* this does the actual multi-iteration blur. We use the gPu buffers we've already set up, we also pass a function pointer to the damping function if the user chooses damping (solution is a little hacky, really just a placeholder so that damping works - it's far too slow with the data being passed back and forth from main memory to/from gPu every iteration). For gauss3x3 and 5x5 kernels it returns 0 as it's been done on gPu. For anything else it returns -1 as I haven't finished transposing those to OpenCl yet. */
	int ret = OpenCLgaussianBlur(tmpI, tmp, gpuSelected, helper, sharpenParam.deconviter, tmpI_mem_obj, tmp_mem_obj, W, H, sigma, lum_mem_obj, false, nullptr, GAUSS_DIV, luminance, damping, &dcdamping);
  if (ret == -1)
    //in this case, we have to transfer the gPu buffers back to CPU and just do the multi iteration blur as normal. This is a placeholder until it's all done on gPu.
	 {
	  JaggedArray<float> tmpI_gPu_mirror(W, H), tmp_gPu_mirror(W, H); 
          helper->gPu_buffer_to_CPU_jagged_array(tmpI_mem_obj, tmpI_gPu_mirror, W, H);
          helper->gPu_buffer_to_CPU_jagged_array(tmp_mem_obj, tmp_gPu_mirror, W, H);
          #ifdef _OPENMP
          #pragma omp parallel
          #endif
          {
	for (int k = 0; k < sharpenParam.deconviter; k++) {
	   if (!needdamp)
	     gaussianBlur(tmpI_gPu_mirror, tmp_gPu_mirror, W, H, sigma, false, nullptr, GAUSS_DIV, luminance); 
           else { gaussianBlur(tmpI_gPu_mirror, tmp_gPu_mirror, W, H, sigma); dcdamping(tmp_gPu_mirror, luminance, damping, W, H); }
	   gaussianBlur(tmp_gPu_mirror, tmpI_gPu_mirror, W, H, sigma, false, nullptr, GAUSS_MULT); 
	  }
           }
	 helper->CPU_jagged_array_to_gPu_buffer(tmpI_gPu_mirror, tmpI_mem_obj, W, H);
         helper->CPU_jagged_array_to_gPu_buffer(tmp_gPu_mirror, tmp_mem_obj, W, H);
	 }
      }

    /* test how long the multi-iteration blur took */
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start_ch;
    std::cout << "Time after multiple gauss: " << elapsed.count() << " s\n";
  
	
    if (OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
      {
	/* this section takes a square chunk of values from both CPU and gPubuffer and compares them.
	   If you turn on debug mode you should see that the results are the same. This is enabled by the "-cl-fp32-correctly-rounded-divide-sqrt" option we give when building our kernels. It's still relatively fast.
	 */
	 float* temp_store_ = new float[W*H]();
	 error_code = clEnqueueReadBuffer(helper->command_queue, tmpI_mem_obj, CL_TRUE, 0, W*H*sizeof(float), temp_store_, 0, nullptr, nullptr);
	 printf("Before is %f, Temp is %f\n\n", tmpI[sampleI][sampleJ], temp_store_[sampleI*W + sampleJ]);
	 for (int i = (H - 3); i < H; i++)
	   {
	     for (int j = (W - 3); j < W; j++)
	     {
	        printf("Difference: CPU is %f, gPu is %f at %d,%d", tmpI[i][j], temp_store_[i*W + j], i, j);
		if (tmpI[i][j] == temp_store_[i*W + j])  printf(" -- CPU and OpenCL results are the same\n");
		else if (fabs(temp_store_[i*W + j] - tmpI[i][j]) < 0.01)  printf("-- Difference is less than one in one hundred\n");	    
	     }
	    }  
	 delete[] temp_store_;

	 printf("\nAfter multi-iteration blur, before intp2: blend result CPU/gPu is %f, %f\n", blend_jagged_array[sampleI][sampleJ],  helper->debug_get_value_from_GPU_buffer(blend_mem_obj, sampleI, sampleJ, W, H));
	 printf("\nAfter multi-iteration blur, before intp2: tmpI result CPU, gPu, gPu Buffer is %f,%f, %f\n", tmpI[sampleI][sampleJ], tmpI[sampleI][sampleJ],  helper->debug_get_value_from_GPU_buffer(tmpI_mem_obj, sampleI, sampleJ, W, H));
	 printf("\nAfter multi-iteration blur, before intp2: luminance result CPU/gPu is %f, %f\n", luminance[sampleI][sampleJ],  helper->debug_get_value_from_GPU_buffer(lum_mem_obj, sampleI, sampleJ, W, H));
       }
         /**** OpenCL section 5 ***************** 
      Transposition of 'intp(blend_jagged_array[i][j] * amount, max(tmpI[i][j], 0.0f), luminance[i][j]);' into OpenCL
     ****************************************/	
    if (OpenCL_helper::OpenCL_usable(gpuSelected) == true_ || OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
      {
	OpenCL_intp_2(helper, W, H, blend_mem_obj, tmpI_mem_obj, lum_mem_obj, &amount);	
      }

      if (OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
	{
    	printf("\ngPu luminance result after second intp is %f\n", helper->debug_get_value_from_GPU_buffer(lum_mem_obj, sampleI, sampleJ, W, H) );
	}

        /**** CPU section 5 - intp  *****************/
      if (OpenCL_helper::OpenCL_usable(gpuSelected) == false_ || OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
      {
       #ifdef _OPENMP
         #pragma omp for
       #endif	      
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
	      luminance[i][j] = intp(blend_jagged_array[i][j] * amount, max(tmpI[i][j], 0.0f), luminance[i][j]);
            }
        }
      }

      
      if (OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
	{
	printf("\nCPU luminance result after second intp is %f\n", luminance[sampleI][sampleJ]);
	}
 

        if (sharpenParam.blurradius >= 0.25) {
            JaggedArray<float> &blur = *blurbuffer;

	   /**** OpenCL section 6 ***************** 
		 Transposition of ' intp(blend_jagged_array[i][j], luminance[i][j], max(blur[i][j], 0.0f))' into OpenCL
	   ****************************************/
	   if (OpenCL_helper::OpenCL_usable(gpuSelected) == true_ || OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
	    OpenCL_intp_3(helper, W, H, blend_mem_obj, lum_mem_obj, blur_mem_obj);
	     
	   if (OpenCL_helper::OpenCL_usable(gpuSelected) == false_ || OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
	      {
	  /**** CPU section 6 - intp *****************/
               #ifdef _OPENMP
                 #pragma omp for
               #endif
		for (int i = 0; i < H; ++i) {
		  for (int j = 0; j < W; ++j) {
                    luminance[i][j] = intp(blend_jagged_array[i][j], luminance[i][j], max(blur[i][j], 0.0f));    
		  }
		}
		printf("\nCPU luminance result after third intp is %f\n", luminance[sampleI][sampleJ]);		
	      }
        }

    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start_ch;
    std::cout << "Time before read back from gPu: " << elapsed.count() << " s\n";
    /* read gPu buffer back into luminance */
    if (OpenCL_helper::OpenCL_usable(gpuSelected) == true_ || OpenCL_helper::OpenCL_usable(gpuSelected) == debug_)
      {
	  error_code = clEnqueueReadBuffer(helper->command_queue, lum_mem_obj, CL_TRUE, 0, W*H*sizeof(float), read_storage, 0, nullptr, nullptr);
	  #ifdef _OPENMP
                #pragma omp for
                #endif
	    for (int i = 0; i < H; i++)
	      {
	        for (int j = 0; j < W; j++)
		  {
		    luminance[i][j] = read_storage[i*W + j];
		  }
	      }

	    printf("\ngPu luminance result after third intp is %f\n", luminance[sampleI][sampleJ]); 
          delete[] read_storage;
    }

    finish = std::chrono::high_resolution_clock::now();
     elapsed = finish - start_ch;
    std::cout << "Time: " << elapsed.count() << " s\n";
	// } // end parallel
    delete blurbuffer;
}

void ImProcFunctions::deconvsharpeningloc (float** luminance, float** tmp, int W, int H, float** loctemp, int damp, double radi, int ite, int amo, int contrast, double blurrad, int sk)
{
    // BENCHFUN

    if (amo < 1) {
        return;
    }
    JaggedArray<float> blend(W, H);
    float contras = contrast / 100.f;
    buildBlendMask(luminance, blend, W, H, contras, 1.f);


    JaggedArray<float> tmpI(W, H);


#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            tmpI[i][j] = max(luminance[i][j], 0.f);
        }
    }

    // calculate contrast based blend factors to reduce sharpening in regions with low contrast

    JaggedArray<float>* blurbuffer = nullptr;

    if (blurrad >= 0.25) {
        blurbuffer = new JaggedArray<float>(W, H);
        JaggedArray<float> &blur = *blurbuffer;
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            gaussianBlur(tmpI, blur, W, H, blurrad);
#ifdef _OPENMP
            #pragma omp for
#endif
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    blur[i][j] = intp(blend[i][j], luminance[i][j], std::max(blur[i][j], 0.0f));
                }
            }
        }
    }

    float damping = (float) damp / 5.0;
    bool needdamp = damp > 0;
    double sigma = radi / sk;
    const float amount = (float) amo / 100.f;

    if (sigma < 0.26f) {
        sigma = 0.26f;
    }

    int itera = ite;

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        for (int k = 0; k < itera; k++) {
            if (!needdamp) {
                // apply gaussian blur and divide luminance by result of gaussian blur
            //    gaussianBlur (tmpI, tmp, W, H, sigma, nullptr, GAUSS_DIV, luminance);
	      gaussianBlur(tmpI, tmp, W, H, sigma, false, nullptr, GAUSS_DIV, luminance);
            } else {
                // apply gaussian blur + damping
                gaussianBlur (tmpI, tmp, W, H, sigma);
                dcdamping (tmp, luminance, damping, W, H);
            }

            gaussianBlur (tmp, tmpI, W, H, sigma, false, nullptr, GAUSS_MULT);
        } // end for


#ifdef _OPENMP
        #pragma omp for
#endif

        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++) {
                loctemp[i][j] = intp(blend[i][j] * amount, max(tmpI[i][j], 0.0f), luminance[i][j]);
            }
            
        if (blurrad >= 0.25) {
            JaggedArray<float> &blur = *blurbuffer;
#ifdef _OPENMP
        #pragma omp for
#endif
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    loctemp[i][j] = intp(blend[i][j], loctemp[i][j], max(blur[i][j], 0.0f));
                }
            }
        }
            
    } // end parallel
    delete blurbuffer;


}


void ImProcFunctions::sharpening (LabImage* lab, const procparams::SharpeningParams &sharpenParam, bool showMask)
{

    if ((!sharpenParam.enabled) || sharpenParam.amount < 1 || lab->W < 8 || lab->H < 8) {
        return;
    }

    int W = lab->W, H = lab->H;

    // calculate contrast based blend factors to reduce sharpening in regions with low contrast
    JaggedArray<float> blend(W, H);
    float contrast = sharpenParam.contrast / 100.0;
    buildBlendMask(lab->L, blend, W, H, contrast);

    if(showMask) {
#ifdef _OPENMP
        #pragma omp parallel for
#endif

        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                lab->L[i][j] = blend[i][j] * 32768.f;
            }
        }
        return;
    }

    JaggedArray<float> b2(W, H);

    if (sharpenParam.method == "rld") {
        auto start_ch= std::chrono::high_resolution_clock::now();
        deconvsharpening(lab->L, b2, blend, lab->W, lab->H, sharpenParam, scale);
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start_ch;
        std::cout << "Overall time: " << elapsed.count() << " s\n";
        return;
    }
BENCHFUN

    // Rest is UNSHARP MASK
    float** b3 = nullptr;

    if (sharpenParam.edgesonly) {
        b3 = new float*[H];

        for (int i = 0; i < H; i++) {
            b3[i] = new float[W];
        }
    }

    JaggedArray<float> blur(W, H);

    if (sharpenParam.blurradius >= 0.25) {
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            gaussianBlur(lab->L, blur, W, H, sharpenParam.blurradius);
#ifdef _OPENMP
            #pragma omp for
#endif
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    blur[i][j] = intp(blend[i][j], lab->L[i][j], std::max(blur[i][j], 0.0f));
                }
            }
        }
    }


#ifdef _OPENMP
    #pragma omp parallel
#endif
    {

        if (!sharpenParam.edgesonly) {
            gaussianBlur (lab->L, b2, W, H, sharpenParam.radius / scale);
        } else {
            bilateral<float, float> (lab->L, (float**)b3, b2, W, H, sharpenParam.edges_radius / scale, sharpenParam.edges_tolerance, multiThread);
            gaussianBlur (b3, b2, W, H, sharpenParam.radius / scale);
        }
    }
    float** base = lab->L;

    if (sharpenParam.edgesonly) {
        base = b3;
    }

    if (!sharpenParam.halocontrol) {
#ifdef _OPENMP
        #pragma omp parallel for
#endif

        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++) {
                constexpr float upperBound = 2000.f;  // WARNING: Duplicated value, it's baaaaaad !
                float diff = base[i][j] - b2[i][j];
                float delta = sharpenParam.threshold.multiply<float, float, float>(
                                  min(fabsf(diff), upperBound),                   // X axis value = absolute value of the difference, truncated to the max value of this field
                                  sharpenParam.amount * diff * 0.01f        // Y axis max value
                              );
                lab->L[i][j] = intp(blend[i][j], lab->L[i][j] + delta, lab->L[i][j]);
            }
    } else {
        if (!sharpenParam.edgesonly) {
            // make a deep copy of lab->L
            JaggedArray<float> labCopy(W, H);

#ifdef _OPENMP
            #pragma omp parallel for
#endif

            for( int i = 0; i < H; i++ )
                for( int j = 0; j < W; j++ ) {
                    labCopy[i][j] = lab->L[i][j];
                }

            sharpenHaloCtrl (lab->L, b2, labCopy, blend, W, H, sharpenParam);
        } else {
            sharpenHaloCtrl (lab->L, b2, base, blend, W, H, sharpenParam);
        }

    }

    if (sharpenParam.edgesonly) {
        for (int i = 0; i < H; i++) {
            delete [] b3[i];
        }

        delete [] b3;
    }

    if (sharpenParam.blurradius >= 0.25) {
#ifdef _OPENMP
    #pragma omp parallel for
#endif
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                lab->L[i][j] = intp(blend[i][j], lab->L[i][j], max(blur[i][j], 0.0f));
            }
        }
    }

}

// To the extent possible under law, Manuel Llorens <manuelllorens@gmail.com>
// has waived all copyright and related or neighboring rights to this work.
// This code is licensed under CC0 v1.0, see license information at
// http://creativecommons.org/publicdomain/zero/1.0/

//! MicroContrast is a sharpening method developed by Manuel Llorens and documented here: http://www.rawness.es/sharpening/?lang=en
//! <BR>The purpose is maximize clarity of the image without creating halo's.
//! <BR>Addition from JD : pyramid  + pondered contrast with matrix 5x5
//! <BR>2017 Ingo Weyrich : reduced processing time
//! \param luminance : Luminance channel of image
void ImProcFunctions::MLmicrocontrast(float** luminance, int W, int H)
{
    if (!params->sharpenMicro.enabled || params->sharpenMicro.contrast == 100 || params->sharpenMicro.amount < 1.0) {
        return;
    }
BENCHFUN
    const int k = params->sharpenMicro.matrix ? 1 : 2;
    // k=2 matrix 5x5  k=1 matrix 3x3
    const int width = W, height = H;
    const int unif = params->sharpenMicro.uniformity;
    const float amount = (k == 1 ? 2.7 : 1.) * params->sharpenMicro.amount / 1500.0; //amount 2000.0 quasi no artifacts ==> 1500 = maximum, after artifacts, 25/9 if 3x3

    if (settings->verbose) {
        printf ("Micro-contrast amount %f\n", static_cast<double>(amount));
        printf ("Micro-contrast uniformity %i\n", unif);
    }

    //modulation uniformity in function of luminance
    const float L98[11] = {0.001f, 0.0015f, 0.002f, 0.004f, 0.006f, 0.008f, 0.01f, 0.03f, 0.05f, 0.1f, 0.1f};
    const float L95[11] = {0.0012f, 0.002f, 0.005f, 0.01f, 0.02f, 0.05f, 0.1f, 0.12f, 0.15f, 0.2f, 0.25f};
    const float L92[11] = {0.01f, 0.015f, 0.02f, 0.06f, 0.10f, 0.13f, 0.17f, 0.25f, 0.3f, 0.32f, 0.35f};
    const float L90[11] = {0.015f, 0.02f, 0.04f, 0.08f, 0.12f, 0.15f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    const float L87[11] = {0.025f, 0.03f, 0.05f, 0.1f, 0.15f, 0.25f, 0.3f, 0.4f, 0.5f, 0.63f, 0.75f};
    const float L83[11] = {0.055f, 0.08f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.75f, 0.85f};
    const float L80[11] = {0.15f, 0.2f, 0.25f, 0.3f, 0.35f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
    const float L75[11] = {0.22f, 0.25f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.85f, 0.9f, 0.95f};
    const float L70[11] = {0.35f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.97f, 1.0f, 1.0f, 1.0f, 1.0f};
    const float L63[11] = {0.55f, 0.6f, 0.7f, 0.8f, 0.85f, 0.9f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const float L58[11] = {0.75f, 0.77f, 0.8f, 0.9f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    //default 5
    //modulation contrast
    const float Cont0[11] = {0.05f, 0.1f, 0.2f, 0.25f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
    const float Cont1[11] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.95f, 1.0f};
    const float Cont2[11] = {0.2f, 0.40f, 0.6f, 0.7f, 0.8f, 0.85f, 0.90f, 0.95f, 1.0f, 1.05f, 1.10f};
    const float Cont3[11] = {0.5f, 0.6f, 0.7f, 0.8f, 0.85f, 0.9f, 1.0f, 1.0f, 1.05f, 1.10f, 1.20f};
    const float Cont4[11] = {0.8f, 0.85f, 0.9f, 0.95f, 1.0f, 1.05f, 1.10f, 1.150f, 1.2f, 1.25f, 1.40f};
    const float Cont5[11] = {1.0f, 1.1f, 1.2f, 1.25f, 1.3f, 1.4f, 1.45f, 1.50f, 1.6f, 1.65f, 1.80f};

    const float sqrt2 = sqrt(2.0);
    const float sqrt1d25 = sqrt(1.25);
    float *LM = new float[width * height]; //allocation for Luminance

    // calculate contrast based blend factors to reduce sharpening in regions with low contrast
    JaggedArray<float> blend(W, H);
    float contrast = params->sharpenMicro.contrast / 100.0;
    buildBlendMask(luminance, blend, W, H, contrast);

#ifdef _OPENMP
    #pragma omp parallel
#endif
{

#ifdef _OPENMP
    #pragma omp for schedule(dynamic,16)
#endif

    for(int j = 0; j < height; j++)
        for(int i = 0, offset = j * width + i; i < width; i++, offset++) {
            LM[offset] = luminance[j][i] / 327.68f; // adjust to [0;100] and to RT variables
        }

#ifdef _OPENMP
    #pragma omp for schedule(dynamic,16)
#endif

    for(int j = k; j < height - k; j++)
        for(int i = k, offset = j * width + i; i < width - k; i++, offset++) {
            float v = LM[offset];

            float contrast;
            if (k == 1) {
                contrast = sqrtf(SQR(LM[offset + 1] - LM[offset - 1]) + SQR(LM[offset + width] - LM[offset - width])) * 0.125f;    //for 3x3
            } else /* if (k==2) */ contrast = sqrtf(SQR(LM[offset + 1] - LM[offset - 1]) + SQR(LM[offset + width] - LM[offset - width])
                                                       + SQR(LM[offset + 2] - LM[offset - 2]) + SQR(LM[offset + 2 * width] - LM[offset - 2 * width])) * 0.0625f; //for 5x5

            contrast = std::min(contrast, 1.f);

            //matrix 5x5
            float temp = v + 4.f *( v * (amount + sqrt2 * amount)); //begin 3x3
            float temp1 = sqrt2 * amount *(LM[offset - width - 1] + LM[offset - width + 1] + LM[offset + width - 1] + LM[offset + width + 1]);
            temp1 += amount * (LM[offset - width] + LM[offset - 1] + LM[offset + 1] + LM[offset + width]);

            temp -= temp1;

            // add JD continue 5x5
            if (k == 2) {
                float temp2 = -(LM[offset + 2 * width] + LM[offset - 2 * width] + LM[offset - 2] + LM[offset + 2]);

                temp2 -= sqrt1d25 * (LM[offset + 2 * width - 1] + LM[offset + 2 * width + 1] + LM[offset +  width + 2] + LM[offset +  width - 2] + 
                                     LM[offset - 2 * width - 1] + LM[offset - 2 * width + 1] + LM[offset -  width + 2] + LM[offset -  width - 2]);

                temp2 -= sqrt2 * (LM[offset + 2 * width - 2] + LM[offset + 2 * width + 2] + LM[offset - 2 * width - 2] + LM[offset - 2 * width + 2]);
                temp2 += 18.601126159f * v ; // 18.601126159 = 4 + 4 * sqrt(2) + 8 * sqrt(1.25)
                temp2 *= 2.f * amount;
                temp += temp2;
            }

            temp = std::max(temp, 0.f);

            for(int row = j - k; row <= j + k; ++row) {
                for(int offset2 = row * width + i - k; offset2 <= row * width + i + k; ++offset2) {
                    if((LM[offset2] - temp) * (v - LM[offset2]) > 0.f) {
                        temp = intp(0.75f, temp, LM[offset2]);
                        goto breakout;
                    }
                }
            }
            breakout:

            if (LM[offset] > 95.0f || LM[offset] < 5.0f) {
                contrast *= Cont0[unif];    //+ JD : luminance  pyramid to adjust contrast by evaluation of LM[offset]
            } else if (LM[offset] > 90.0f || LM[offset] < 10.0f) {
                contrast *= Cont1[unif];
            } else if (LM[offset] > 80.0f || LM[offset] < 20.0f) {
                contrast *= Cont2[unif];
            } else if (LM[offset] > 70.0f || LM[offset] < 30.0f) {
                contrast *= Cont3[unif];
            } else if (LM[offset] > 60.0f || LM[offset] < 40.0f) {
                contrast *= Cont4[unif];
            } else {
                contrast *= Cont5[unif];    //(2.0f/k)*Cont5[unif];
            }

            contrast = std::min(contrast, 1.f);

            float tempL = intp(contrast, LM[offset], temp);
            // JD: modulation of microcontrast in function of original Luminance and modulation of luminance
            if (tempL > LM[offset]) {
                float temp2 = tempL / LM[offset]; //for highlights
                temp2 = std::min(temp2, 1.7f); //limit action
                temp2 -= 1.f;
                if (LM[offset] > 98.0f) {
                    temp = 0.f;
                } else if (LM[offset] > 95.0f) {
                    temp = L95[unif];
                } else if (LM[offset] > 92.0f) {
                    temp = L92[unif];
                } else if (LM[offset] > 90.0f) {
                    temp = L90[unif];
                } else if (LM[offset] > 87.0f) {
                    temp = L87[unif];
                } else if (LM[offset] > 83.0f) {
                    temp = L83[unif];
                } else if (LM[offset] > 80.0f) {
                    temp = L80[unif];
                } else if (LM[offset] > 75.0f) {
                    temp = L75[unif];
                } else if (LM[offset] > 70.0f) {
                    temp = L70[unif];
                } else if (LM[offset] > 63.0f) {
                    temp = L63[unif];
                } else if (LM[offset] > 58.0f) {
                    temp = L58[unif];
                } else if (LM[offset] > 42.0f) {
                    temp = L58[unif];
                } else if (LM[offset] > 37.0f) {
                    temp = L63[unif];
                } else if (LM[offset] > 30.0f) {
                    temp = L70[unif];
                } else if (LM[offset] > 25.0f) {
                    temp = L75[unif];
                } else if (LM[offset] > 20.0f) {
                    temp = L80[unif];
                } else if (LM[offset] > 17.0f) {
                    temp = L83[unif];
                } else if (LM[offset] > 13.0f) {
                    temp = L87[unif];
                } else if (LM[offset] > 10.0f) {
                    temp = L90[unif];
                } else if (LM[offset] > 5.0f) {
                    temp = L95[unif];
                } else {
                    temp = 0.f;
                }
                luminance[j][i] = intp(blend[j][i], luminance[j][i] * (temp * temp2 + 1.f), luminance[j][i]);
            } else {

                float temp4 = LM[offset] / tempL; //

                if (temp4 > 1.0f) {
                    temp4 = std::min(temp4, 1.7f); //limit action
                    temp4 -= 1.f;
                    if (LM[offset] < 2.0f) {
                        temp = L98[unif];
                    } else if (LM[offset] < 5.0f) {
                        temp = L95[unif];
                    } else if (LM[offset] < 8.0f) {
                        temp = L92[unif];
                    } else if (LM[offset] < 10.0f) {
                        temp = L90[unif];
                    } else if (LM[offset] < 13.0f) {
                        temp = L87[unif];
                    } else if (LM[offset] < 17.0f) {
                        temp = L83[unif];
                    } else if (LM[offset] < 20.0f) {
                        temp = L80[unif];
                    } else if (LM[offset] < 25.0f) {
                        temp = L75[unif];
                    } else if (LM[offset] < 30.0f) {
                        temp = L70[unif];
                    } else if (LM[offset] < 37.0f) {
                        temp = L63[unif];
                    } else if (LM[offset] < 42.0f) {
                        temp = L58[unif];
                    } else if (LM[offset] < 58.0f) {
                        temp = L58[unif];
                    } else if (LM[offset] < 63.0f) {
                        temp = L63[unif];
                    } else if (LM[offset] < 70.0f) {
                        temp = L70[unif];
                    } else if (LM[offset] < 75.0f) {
                        temp = L75[unif];
                    } else if (LM[offset] < 80.0f) {
                        temp = L80[unif];
                    } else if (LM[offset] < 83.0f) {
                        temp = L83[unif];
                    } else if (LM[offset] < 87.0f) {
                        temp = L87[unif];
                    } else if (LM[offset] < 90.0f) {
                        temp = L90[unif];
                    } else if (LM[offset] < 95.0f) {
                        temp = L95[unif];
                    } else {
                        temp = 0.f;
                    }
                    luminance[j][i] = intp(blend[j][i], luminance[j][i] / (temp * temp4 + 1.f), luminance[j][i]);
                }
            }
        }
}
    delete [] LM;
}

void ImProcFunctions::MLmicrocontrast(LabImage* lab)
{
    MLmicrocontrast(lab->L, lab->W, lab->H);
}

void ImProcFunctions::MLmicrocontrastcam(CieImage* ncie)
{
    MLmicrocontrast(ncie->sh_p, ncie->W, ncie->H);
}

void ImProcFunctions::sharpeningcam (CieImage* ncie, float** b2, bool showMask)
{
    if ((!params->sharpening.enabled) || params->sharpening.amount < 1 || ncie->W < 8 || ncie->H < 8) {
        return;
    }

    int W = ncie->W, H = ncie->H;

    // calculate contrast based blend factors to reduce sharpening in regions with low contrast
    JaggedArray<float> blend(W, H);
    float contrast = params->sharpening.contrast / 100.0;
    buildBlendMask(ncie->sh_p, blend, W, H, contrast);
    if(showMask) {
#ifdef _OPENMP
        #pragma omp parallel for
#endif

        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                ncie->sh_p[i][j] = blend[i][j] * 32768.f;
            }
        }
        return;
    }

    if (params->sharpening.method == "rld") {
        deconvsharpening (ncie->sh_p, b2, blend, ncie->W, ncie->H, params->sharpening, scale);
        return;
    }

    // Rest is UNSHARP MASK

    float** b3 = nullptr;

    if (params->sharpening.edgesonly) {
        b3 = new float*[H];

        for (int i = 0; i < H; i++) {
            b3[i] = new float[W];
        }
    }

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {

        if (!params->sharpening.edgesonly) {
            gaussianBlur (ncie->sh_p, b2, W, H, params->sharpening.radius / scale);
        } else {
            bilateral<float, float> (ncie->sh_p, (float**)b3, b2, W, H, params->sharpening.edges_radius / scale, params->sharpening.edges_tolerance, multiThread);
            gaussianBlur (b3, b2, W, H, params->sharpening.radius / scale);
        }
    }

    float** base = ncie->sh_p;

    if (params->sharpening.edgesonly) {
        base = b3;
    }

    if (!params->sharpening.halocontrol) {
#ifdef _OPENMP
        #pragma omp parallel for
#endif

        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++) {
                constexpr float upperBound = 2000.f;  // WARNING: Duplicated value, it's baaaaaad !
                float diff = base[i][j] - b2[i][j];
                float delta = params->sharpening.threshold.multiply<float, float, float>(
                                  min(fabsf(diff), upperBound),                   // X axis value = absolute value of the difference, truncated to the max value of this field
                                  params->sharpening.amount * diff * 0.01f      // Y axis max value
                              );

                if(ncie->J_p[i][j] > 8.0f && ncie->J_p[i][j] < 92.0f) {
                    ncie->sh_p[i][j] = intp(blend[i][j], ncie->sh_p[i][j] + delta, ncie->sh_p[i][j]);
                }
            }
    } else {
        float** ncieCopy = nullptr;

        if (!params->sharpening.edgesonly) {
            // make deep copy of ncie->sh_p
            ncieCopy = new float*[H];

            for( int i = 0; i < H; i++ ) {
                ncieCopy[i] = new float[W];
            }

#ifdef _OPENMP
            #pragma omp parallel for
#endif

            for( int i = 0; i < H; i++ )
                for( int j = 0; j < W; j++ ) {
                    ncieCopy[i][j] = ncie->sh_p[i][j];
                }

            base = ncieCopy;
        }

        sharpenHaloCtrl (ncie->sh_p, b2, base, blend, W, H, params->sharpening);

        if(ncieCopy) {
            for( int i = 0; i < H; i++ ) {
                delete[] ncieCopy[i];
            }

            delete[] ncieCopy;
        }
    }

    if (params->sharpening.edgesonly) {
        for (int i = 0; i < H; i++) {
            delete [] b3[i];
        }

        delete [] b3;
    }
}

}
