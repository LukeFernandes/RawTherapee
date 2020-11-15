
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
 *  along with RawTherapee.  If not, see <http://www.gnu.org/licenses/>.
 */

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <vector>
#include <map>
#include <iostream>
#include "jaggedarray.h"
#include "../rtgui/options.h"

#include "stdio.h"

#ifndef OpenCL_support
#define OpenCL_support

typedef enum {
	      maxkernel = 5,
	      gauss3x3std = 34,
	      gauss3x3stdnew = 35,
	      gauss3x3div = 23,
	      gauss3x3divnew = 24,
	      gauss3x3mult = 87,
	      gauss3x3mul = 51,
	      intptag = 2,
	      intptag2 = 3,
	      intptag3 = 4,
	      damping = 98
} kernel_tag;

typedef enum {
	      luminance_ = 1,
	      returned_ = 2	      
} buffer_tag;

typedef struct {
  cl_kernel kernel;
  kernel_tag tag;
} kernel_with_tag;

typedef struct {
  cl_mem mem;
  int W; //we need W and H stored here as we need to chec
  int H;
} bufferWithDims;

typedef enum {
	      true_,
	      false_,
	      debug_
} OpenCL_use;
  

 class OpenCL_helper{
  private:
    static int OpenCL_available_;
    int max_local_item_size = 0;
  public:
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_device_id device_id = nullptr;  
   
   std::map<std::string, cl_kernel> kernels;
   std::map<std::string, bufferWithDims> buffer_set;
   OpenCL_helper();
   cl_kernel setup_kernel(const char* kernel_filename, const char *kernel_name);
   void getLocalWorkGroupSize(size_t* worksize);
   cl_kernel reuse_or_create_kernel(std::string string, const char *kernel_filename, const char *kernel_name);
   cl_mem reuse_or_create_buffer( std::string string, int W, int H, cl_mem_flags flag, float* optionaldata = nullptr); 
   static void JaggedArray_to_1d_array(float* d1_array, rtengine::JaggedArray<float> *jaggedarray, int W, int H);
   static void ArrayofArrays_to_1d_array(float* d1_array, float** d2_array, int W, int H);
   static void d1_array_to_JaggedArray(float* d1_array, rtengine::JaggedArray<float>& jaggedarray, int W, int H);
   static void d1_array_to_2d_array(float* d1_array, float** d2_array, int W, int H);
   void gPu_buffer_to_CPU_jagged_array(cl_mem gpuBuffer, rtengine::JaggedArray<float>& jaggedarray, int W, int H);
   void CPU_jagged_array_to_gPu_buffer(rtengine::JaggedArray<float>& jaggedarray, cl_mem gpuBuffer, int W, int H);
   float debug_get_value_from_GPU_buffer(cl_mem buffer, int X, int Y, int W, int H);
   static bool OpenCL_available();
   static OpenCL_use OpenCL_usable(OpenCL_use user_selection);
  };
#endif

