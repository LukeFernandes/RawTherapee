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
#include <vector>

#define MAX_SOURCE_SIZE (0x100000)

int OpenCL_helper::OpenCL_available_ = 0;

//std::map<std::string, cl_mem> OpenCL_helper::buffer_set;

OpenCL_helper::OpenCL_helper() { 
   
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int error_code = 0;

    error_code = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    printf("Get Platform error code is (0 is success):%d\n", error_code);
     if ( CL_SUCCESS == error_code)
      {
	error_code = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	     printf("Get Device error code is (0 is success):%d\n", error_code);
	if (CL_SUCCESS == error_code)
	  {
	    fflush(stderr);

	    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &error_code);
	    printf("OpenCL Context Error code 1 (0 is success):%d\n", error_code);
	    command_queue = clCreateCommandQueue(context, device_id, 0, &error_code);
	    printf("OpenCL Command Queue Error code 2 (0 is success):%d\n", error_code);
	      // kernels.reserve(MAX_NO_KERNELS);
	      program = NULL;
	  }	
  }
     else {
       //printf("OpenCL not available on host platform.\n");
       OpenCL_available_ = false_;

     }
     //printf("OpenCL helper class created. \n");
 }

cl_kernel  OpenCL_helper::setup_kernel(const char *kernel_filename, const char *kernel_name) {
            FILE *fp;
            char *source_str;
            size_t source_size;
	    cl_int error_code = 0;
	    char full_kernel_filename[80];
	    strcpy(full_kernel_filename, "C:\\code\\repo-rt\\clkernels\\");
	    strcat(full_kernel_filename, kernel_filename);
	    fprintf(stderr, full_kernel_filename);
	    //size_t a = strlen(full_kernel_filename);

            fp = fopen(full_kernel_filename, "r");
                if (!fp) {
	        printf("Failed to load kernel.\n");
                   exit(1);
                         }
    
            source_str = (char*)malloc(MAX_SOURCE_SIZE);
            source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
            fclose( fp );

	    
            program = clCreateProgramWithSource(context, 1, const_cast<const char**>(&source_str), (const size_t *)&source_size, &error_code);
	    if (error_code != 0) printf("Setup kernel error codes:%d,", error_code);
	    error_code = clBuildProgram(program, 1, &device_id, "-cl-fast-relaxed-math", NULL, NULL); //"-cl-fp32-correctly-rounded-divide-sqrt -cl-opt-disable" "-cl-fast-relaxed-math"
	    if (error_code != 0) printf("%d,", error_code);
	     cl_kernel kernel; 
	     kernel = clCreateKernel(program, kernel_name, &error_code); 
	    if (error_code != 0) printf("%d\n", error_code);
            free(source_str);
	    return kernel;

  }

void OpenCL_helper::getLocalWorkGroupSize(size_t* worksize)
     {
       cl_device_id* devices = &device_id;
       if (devices != nullptr) {
       clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), (void*)worksize, NULL);
       }
       else  printf("No device id; cannot get information about device \n");
     }

cl_kernel OpenCL_helper::reuse_or_create_kernel(std::string string, const char *kernel_filename, const char *kernel_name ) {
    cl_kernel kernel;

    //printf("Kernels length is %d \n", (int)this->kernels.size());

    auto search =  kernels.find(string);

    if (search == kernels.end())   // element not found
      {
        kernel = this->setup_kernel(kernel_filename, kernel_name);
	kernels.insert({string, kernel});
	//std::cout << "OpenCL new kernel of type " << string << " created \n";
      }
    else
      {
	kernel = search->second;
	//std::cout << "OpenCL Old kernel of type " << string << " reused \n"; 
      }
    return kernel;
}

cl_mem OpenCL_helper::reuse_or_create_buffer(std::string string, int W, int H, cl_mem_flags flag, float* optionaldata) {
  if (string == "")
    throw "Buffer slot is a null pointer!";
  int error_code = 200;
  cl_mem input_buffer;
  auto search =  buffer_set.find(string);
   if ( (search != buffer_set.end()) && (search->second.W == W) && (search->second.H == H) ) {
		      	input_buffer = search->second.mem;
			 if (optionaldata != nullptr) {
			    error_code = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0, W*H*sizeof(float), optionaldata, 0, NULL, NULL);
			    if (error_code == 0)  std::cout << "\nThe old buffer has been rewritten successfully\n";
			    else std::cout << "\nOpenCL error " << error_code;
			  }
			 // std::cout << "OpenCL Old memory object " << string << " reused\n";
	       }
   else
     {
			  input_buffer = clCreateBuffer(context, flag, W*H*sizeof(float), NULL, &error_code);
			  buffer_set.insert({string, bufferWithDims{input_buffer, W, H}});
			 		     
			  //std::cout << "New buffer " << string << " created and stored; OpenCL Error code (0 is success): " << error_code << "\n"; 
			  if (optionaldata != nullptr) {
			    error_code = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0, W*H*sizeof(float), optionaldata, 0, NULL, NULL);
			    
			    if (error_code == 0)  std::cout << "\nThe new buffer has been written successfully\n";
			    else std::cout << "\nOpenCL error " << error_code << " with writing data to " << string << " buffer\n";
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

void OpenCL_helper::d1_array_to_JaggedArray(float* d1_array, rtengine::JaggedArray<float>& jaggedarray, int W, int H)
{
  
    for (int i = 0; i < H; i++)
	       {
	        for (int j = 0; j < W; j++)
	         {
		   jaggedarray[i][j] = d1_array[i*W + j];
	         }
            	}
}

void OpenCL_helper::d1_array_to_2d_array(float* d1_array, float** d2_array, int W, int H)
{
  
 for (int i = 0; i < H; i++)
	       {
	        for (int j = 0; j < W; j++)
	         {
		   d2_array[i][j] = d1_array[i*W + j];
	         }
            	}

}

void OpenCL_helper::gPu_buffer_to_CPU_jagged_array(cl_mem gpuBuffer, rtengine::JaggedArray<float>& jaggedarray, int W, int H)
{
   int error_code;
   float* temp = new float[W*H]();
   clEnqueueReadBuffer(command_queue, gpuBuffer, CL_TRUE, 0, W*H*sizeof(float),temp, 0, nullptr, nullptr);
   OpenCL_helper::d1_array_to_JaggedArray(temp, jaggedarray, W, H);
   delete [] temp;
}

void OpenCL_helper::CPU_jagged_array_to_gPu_buffer(rtengine::JaggedArray<float>& jaggedarray, cl_mem gpuBuffer, int W, int H)
{
   int error_code;
   float* temp = new float[W*H]();
   OpenCL_helper::JaggedArray_to_1d_array(temp, &jaggedarray, W, H);
   error_code = clEnqueueWriteBuffer(command_queue, gpuBuffer, CL_TRUE, 0, W*H*sizeof(float), temp, 0, nullptr, nullptr);
   delete [] temp;
}


float OpenCL_helper::debug_get_value_from_GPU_buffer(cl_mem buffer, int Y, int X, int W, int H) {

      float* temp_store = (float*)malloc(W*H*sizeof(float));
      int error_code = 0;
      error_code = clEnqueueReadBuffer(command_queue, buffer, CL_TRUE, 0, W*H*sizeof(float), temp_store, 0, nullptr, nullptr);
      float ret_value = temp_store[W*Y + X];
      free(temp_store);
      return ret_value;

}

bool OpenCL_helper::OpenCL_available() {
  
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_platforms;
  if (OpenCL_available_ == 1) 
  {
    return false;

  }
  else if (OpenCL_available_ == 2) 
  {
    return true;
  }
 else if (OpenCL_available_ == 0) 
  {
  if (CL_SUCCESS == clGetPlatformIDs(1, &platform_id, &ret_num_platforms)) {
	OpenCL_available_ = 2;
        return true;
      }
      else {
	OpenCL_available_ = 1;
	return false;
      }
  }
 else return false;
}
   

OpenCL_use OpenCL_helper::OpenCL_usable(OpenCL_use user_selection) {
  
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_platforms;
      if ((OpenCL_available() == true)  && (user_selection == true_)) {
        return true_;
      }
      else if (user_selection == false_) {
	return false_;
      }
      else if (user_selection == debug_) {
	return debug_;
      }
      else return false_;
}
   
