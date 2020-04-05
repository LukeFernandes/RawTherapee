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
#pragma once
#include "StopWatch.h"
#include "opthelper.h"
#include "OpenCL_support.h"

enum eGaussType {GAUSS_STANDARD, GAUSS_MULT, GAUSS_DIV};

typedef struct {
  double c0;
  double c1;
  double c2;
  double b0;
  double b1;
  //bool already_done_on_CPU;
} reprocess_data;

void gaussianBlur(float** src, float** dst, const int W, const int H, const double sigma, bool useBoxBlur = false, float *buffer = nullptr, eGaussType gausstype = GAUSS_STANDARD, float** buffer2 = nullptr);

void OpenCLgaussianBlur(OpenCL_helper* helper, int iterations, cl_mem src_clmem, cl_mem dst_clmem, cl_mem div_clmem, float** src, float** dst, const int W, const int H, const double sigma, bool useBoxBlur = false, float *buffer = nullptr, eGaussType gausstype = GAUSS_STANDARD, float** buffer2 = nullptr, float damping = 0.0f);

