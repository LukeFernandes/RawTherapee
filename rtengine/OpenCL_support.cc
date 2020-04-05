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
//#include <iterator>

#include "OpenCL_support.h"
#include <algorithm>

#define MAX_SOURCE_SIZE (0x100000)
#define MAX_NO_KERNELS (10)


OpenCL_helper::OpenCL_helper() { 
   
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int error_code = NULL;

     if ( CL_SUCCESS ==  clGetPlatformIDs(1, &platform_id, &ret_num_platforms) )
      {
	fprintf(stderr, "Checkpoint general new1 reached\n");
	fflush(stderr);
	if (CL_SUCCESS == clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices))
	  {
	    fprintf(stderr, "Checkpoint general new2 reached\n");
	    fflush(stderr);
	    cl_int error_code = NULL;
	    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &error_code);
	     fprintf(stderr, "OpenCL general Error code 1 (0 is success):%d\n", error_code);
             fflush(stderr);
	    command_queue = clCreateCommandQueue(context, device_id, 0, &error_code);
	     fprintf(stderr, "OpenCL general Error code 2 (0 is success):%d\n", error_code);
              fflush(stderr);
	      kernels.reserve(MAX_NO_KERNELS);
	      program = NULL;
	  }	
  }
      fprintf(stderr, "OpenCL helper class created. \n");
      fflush(stderr);
 }

  cl_kernel*  OpenCL_helper::setup_kernel(const char *kernel_filename, const char *kernel_name, kernel_tag flag) {
            FILE *fp;
            char *source_str;
            size_t source_size;
	    cl_int error_code = NULL;
	    char full_kernel_filename[80];
	    strcpy(full_kernel_filename, "C:\\code\\repo-rt\\clkernels\\");
	    strcat(full_kernel_filename, kernel_filename);
	    fprintf(stderr, full_kernel_filename);
	    size_t a = strlen(full_kernel_filename);

            fp = fopen(full_kernel_filename, "r");
                if (!fp) {
	        fprintf(stderr, "Failed to load kernel.\n");
         	fflush(stderr);
                   exit(1);
                         }
    
            source_str = (char*)malloc(MAX_SOURCE_SIZE);
            source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
            fclose( fp );

            fprintf(stderr, "Checkpoint SK1A reached. \n");
            fflush(stderr);

	    
            program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &error_code);
	     fprintf(stderr, "Setup kernel error codes:%d,", error_code);
             fflush(stderr);
	    error_code = clBuildProgram(program, 1, &device_id, "-cl-fp32-correctly-rounded-divide-sqrt -cl-opt-disable", NULL, NULL);
	     fprintf(stderr, "%d,", error_code);
             fflush(stderr);
	     cl_kernel *kernel = (cl_kernel*)malloc(sizeof(cl_kernel));
	     *kernel = clCreateKernel(program, kernel_name, &error_code); 
	     kernels.push_back(kernel_with_tag{*kernel, flag});
	     fprintf(stderr, "%d\n", error_code);
             fflush(stderr);
            free(source_str);
	    return kernel;

  }

cl_kernel OpenCL_helper::reuse_or_create_kernel(kernel_tag desired_tag, const char *kernel_filename, const char *kernel_name ) {
      cl_kernel kernel;
      int error_code = 0;

      fprintf(stderr, "Kernels length is %d \n", (int)this->kernels.size()); fflush(stderr);
      std::vector<kernel_with_tag>::iterator it = std::find_if(this->kernels.begin(), this->kernels.end(), [&desired_tag](const kernel_with_tag& obj) {return obj.tag == desired_tag;});

    if (it != this->kernels.end())   // element not found
      {
	kernel = it->kernel;
	fprintf(stderr, "OpenCL Old kernel of type %d reused mem\n", (int)desired_tag);  fflush(stderr);
	return kernel;
      }
    else
      {
        kernel = *(this->setup_kernel(kernel_filename, kernel_name, desired_tag));
	fprintf(stderr, "OpenCL new kernel of type %d created mem\n", (int)desired_tag);  fflush(stderr);
	return kernel;
      }	
}

cl_mem OpenCL_helper::reuse_or_create_buffer(cl_mem* buffer_slot, int W, int H, cl_mem_flags flag, float* optionaldata) {
  if (buffer_slot == NULL || nullptr)
    throw "Buffer slot is a null pointer!";
  int error_code;
  cl_mem input_buffer;
  if (*buffer_slot != nullptr) {
		      	input_buffer = *buffer_slot;
			fprintf(stderr, "OpenCL Old memory object reused\n"); fflush(stderr);
		      }
		      else
			{
			  input_buffer = clCreateBuffer(context, flag, W*H*sizeof(float), NULL, &error_code);
			  *buffer_slot = input_buffer;
			  fprintf(stderr, "New buffer created and stored; OpenCL Error code (0 is success):%d\n", error_code); 
			  if (optionaldata != nullptr) {
			    int error_code = 1;
			    error_code = clEnqueueWriteBuffer(command_queue, *buffer_slot, CL_TRUE, 0, W*H*sizeof(float), optionaldata, 0, NULL, NULL);
			    if (error_code == 0) fprintf(stderr, "\nThe new buffer has been written successfully\n");
			    else fprintf(stderr, "\nOpenCL error %d", error_code);
			    fflush(stderr);
			  }
			}
       return input_buffer;
}


void  OpenCL_helper::JaggedArray_to_1d_array(float* d1_array, rtengine::JaggedArray<float> *jaggedarray, int W, int H) {

    for (int i = 0; i < H; i++)
	       {
	        for (int j = 0; j < W; j++)
	         {
		   d1_array[i*W + j] = (*jaggedarray)[i][j];
	         }
            	}
}

void OpenCL_helper::ArrayofArrays_to_1d_array(float* d1_array, float** d2_array, int W, int H)
{

    for (int i = 0; i < H; i++)
	       {
	        for (int j = 0; j < W; j++)
	         {
		   d1_array[i*W + j] = d2_array[i][j];
	         }
            	}

}

void OpenCL_helper::d1_array_to_JaggedArray(float* d1_array, rtengine::JaggedArray<float> *jaggedarray, int W, int H)
{
  
    for (int i = 0; i < H; i++)
	       {
	        for (int j = 0; j < W; j++)
	         {
		   (*jaggedarray)[i][j] = d1_array[i*W + j];
	         }
            	}

}
   
