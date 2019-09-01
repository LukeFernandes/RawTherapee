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

#include <CL/cl.h>
#include <vector>
#include "jaggedarray.h"

#include "stdio.h"

#ifndef OpenCL_support
#define OpenCL_support

typedef enum {
	      maxkernel = 5,
	      gauss3x3std = 34,
	      gauss3x3div = 23,
	      gauss3x3mult = 87,
	      intptag = 2,
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
  buffer_tag tag;
} buffer_with_tag;
  
  

 class OpenCL_helper{ 
  public:
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_device_id device_id = NULL;
    cl_mem luminance_ = NULL;
    cl_mem tmpI_ = NULL;
    cl_mem blur_ = NULL;
    cl_mem blend_ = NULL;
   cl_mem src_ = NULL;
   cl_mem oldsrc_ = NULL;
   cl_mem olddst_ = NULL;
   cl_mem newdst_ = NULL;
   cl_mem gaussret_ = NULL;
   cl_mem X_ = NULL;
   cl_mem Y_ = NULL;
   cl_mem div_ = NULL;
   cl_mem dampfac_ = NULL;
   

    std::vector<kernel_with_tag> kernels;
    std::vector<buffer_with_tag> buffers;
   OpenCL_helper();
   cl_kernel* setup_kernel(const char* kernel_filename, const char *kernel_name, kernel_tag flag = (kernel_tag)0);
   cl_kernel  reuse_or_create_kernel(kernel_tag desired_tag, const char *kernel_filename, const char *kernel_name);
   cl_mem reuse_or_create_buffer(cl_mem buffer_slot, int W, int H, cl_mem_flags flag); 
   void JaggedArray_to_1d_array(float* d1_array, rtengine::JaggedArray<float> *jaggedarray, int W, int H);
   static void ArrayofArrays_to_1d_array(float* d1_array, float** d2_array, int W, int H);
  };
#endif
