/*
 *  This file is part of RawTherapee.
 *
 *  Copyright (c) 2019 rom9
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
#include <cmath>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "rawimagesource.h"

#include "mytime.h"
#include "opthelper.h"
#include "procparams.h"
#include "rt_algo.h"
#include "rtengine.h"

//#define BENCHMARK
#include "StopWatch.h"

namespace rtengine
{

extern const Settings* settings;

}

namespace
{

bool channelsAvg(
    const rtengine::RawImage* ri,
    int width,
    int height,
    array2D<float>& rawData,
    rtengine::Coord spotPos,
    int spotSize,
    const rtengine::procparams::FilmNegativeParams& params,
    std::array<float, 3>& avgs
)
{
    avgs = {}; // Channel averages

    if (ri->getSensorType() != rtengine::ST_BAYER && ri->getSensorType() != rtengine::ST_FUJI_XTRANS) {
        return false;
    }

    if (rtengine::settings->verbose) {
        printf("Spot coord:  x=%d y=%d\n", spotPos.x, spotPos.y);
    }

    const int half_spot_size = spotSize / 2;

    const int& x1 = spotPos.x - half_spot_size;
    const int& x2 = spotPos.x + half_spot_size;
    const int& y1 = spotPos.y - half_spot_size;
    const int& y2 = spotPos.y + half_spot_size;

    if (x1 < 0 || x2 > width || y1 < 0 || y2 > height) {
        return false; // Spot goes outside bounds, bail out.
    }

    std::array<int, 3> pxCount = {}; // Per-channel sample counts
    for (int c = spotPos.x - spotSize; c < spotPos.x + spotSize; ++c) {
        for (int r = spotPos.y - spotSize; r < spotPos.y + spotSize; ++r) {
            const int ch = ri->getSensorType() == rtengine::ST_BAYER ? ri->FC(r,c) : ri->XTRANSFC(r,c);

            ++pxCount[ch];

            // If film negative is currently enabled, undo the effect by elevating to 1/exp,
            // in order to sample the original, linear value
            if (params.enabled) {
                avgs[ch] += powf(rawData[r][c], -1.f / (ch == 0 ? params.redExp : ch == 1 ? params.greenExp : params.blueExp));
            } else {
                avgs[ch] += rawData[r][c];
            }
        }
    }

    for (int ch = 0; ch < 3; ++ch) {
        avgs[ch] /= pxCount[ch];
    }

    return true;
}

}

bool rtengine::RawImageSource::getFilmNegativeExponents(Coord2D spotA, Coord2D spotB, int tran, const FilmNegativeParams &currentParams, std::array<float, 3>& newExps)
{
    newExps = {
        static_cast<float>(currentParams.redExp),
        static_cast<float>(currentParams.greenExp),
        static_cast<float>(currentParams.blueExp)
    };

    constexpr int spotSize = 32; // TODO: Make this configurable?

    Coord spot;
    std::array<float, 3> clearVals;
    std::array<float, 3> denseVals;

    // Sample first spot
    transformPosition(spotA.x, spotA.y, tran, spot.x, spot.y);
    if (!channelsAvg(ri, W, H, rawData, spot, spotSize, currentParams, clearVals)) {
        return false;
    }

    // Sample second spot
    transformPosition(spotB.x, spotB.y, tran, spot.x, spot.y);
    if (!channelsAvg(ri, W, H, rawData, spot, spotSize, currentParams, denseVals)) {
        return false;
    }

    // Detect which one is the dense spot, based on green channel
    if (clearVals[1] < denseVals[1]) {
      std::swap(clearVals, denseVals);
    }

    if (settings->verbose) {
        printf("Clear film values: R=%g G=%g B=%g\n", clearVals[0], clearVals[1], clearVals[2]);
        printf("Dense film values: R=%g G=%g B=%g\n", denseVals[0], denseVals[1], denseVals[2]);
    }

    const float denseGreenRatio = clearVals[1] / denseVals[1];

    // Calculate logarithms in arbitrary base
    const auto logBase =
        [](float base, float num) -> float
        {
            return std::log(num) / std::log(base);
        };

    // Calculate exponents for each channel, based on the ratio between the bright and dark values,
    // compared to the ratio in the reference channel (green)
    for (int ch = 0; ch < 3; ++ch) {
        if (ch == 1) {
            newExps[ch] = 2.f;  // Green is the reference channel
        } else {
            newExps[ch] = CLAMP(2.f * logBase(clearVals[ch] / denseVals[ch], denseGreenRatio), 0.3f, 6.f);
        }
    }

    if (settings->verbose) {
        printf("New exponents:  R=%g G=%g B=%g\n", newExps[0], newExps[1], newExps[2]);
    }

    return true;
}

void rtengine::RawImageSource::filmNegativeProcess(const procparams::FilmNegativeParams &params)
{
//    BENCHFUNMICRO

    if (!params.enabled) {
        return;
    }

    const std::array<float, 3> exps = {
        static_cast<float>(params.redExp),
        static_cast<float>(params.greenExp),
        static_cast<float>(params.blueExp)
    };
    
    MyTime t1, t2, t3,t4, t5;

    t1.set();

    // Channel vectors to calculate medians
    std::array<std::vector<float>, 3> cvs;

    // Sample one every 5 pixels, and push the value in the appropriate channel vector.
    // Choose an odd step, not a multiple of the CFA size, to get a chance to visit each channel.
    if (ri->getSensorType() == ST_BAYER) {
        for (int row = 0; row < H; row += 5) {
            const int c0 = ri->FC(row, 0);
            const int c1 = ri->FC(row, 5);
            int col = 0;
            for (; col < W - 5; col += 10) {
                cvs[c0].push_back(rawData[row][col]);
                cvs[c1].push_back(rawData[row][col + 5]);
            }
            if (col < W) {
                cvs[c0].push_back(rawData[row][col]);
            }
        }
    }
    else if (ri->getSensorType() == ST_FUJI_XTRANS) {
        for (int row = 0; row < H; row += 5) {
            const std::array<unsigned int, 6> cs = {
                ri->XTRANSFC(row, 0),
                ri->XTRANSFC(row, 5),
                ri->XTRANSFC(row, 10),
                ri->XTRANSFC(row, 15),
                ri->XTRANSFC(row, 20),
                ri->XTRANSFC(row, 25)
            };
            int col = 0;
            for (; col < W - 25; col += 30) {
                for (int c = 0; c < 6; ++c) {
                    cvs[cs[c]].push_back(rawData[row][col + c * 5]);
                }
            }
            for (int c = 0; col < W; col += 5, ++c) {
                cvs[cs[c]].push_back(rawData[row][col]);
            }
        }
    }

    constexpr float MAX_OUT_VALUE = 65000.f;

    t2.set();

    if (settings->verbose) {
        printf("Median vector fill loop time us: %d\n", t2.etime(t1));
    }

    t2.set();

    std::array<float, 3> medians; // Channel median values
    std::array<float, 3> mults = {
        1.f,
        1.f,
        1.f
    };  // Channel normalization multipliers

    for (int c = 0; c < 3; ++c) {
        // Find median values for each channel
        if (!cvs[c].empty()) {
            findMinMaxPercentile(cvs[c].data(), cvs[c].size(), 0.5f, medians[c], 0.5f, medians[c], true);
            medians[c] = pow_F(rtengine::max(medians[c], 1.f), -exps[c]);
            // Determine the channel multipler so that N times the median becomes 65k. This clips away
            // the values in the dark border surrounding the negative (due to the film holder, for example),
            // the reciprocal of which have blown up to stellar values.
            mults[c] = MAX_OUT_VALUE / (medians[c] * 24.f);
        }
    }

    t3.set();

    if (settings->verbose) {
        printf("Sample count: %zu, %zu, %zu\n", cvs[0].size(), cvs[1].size(), cvs[2].size());
        printf("Medians: %g %g %g\n", medians[0], medians[1], medians[2] );
        printf("Computed multipliers: %g %g %g\n", mults[0], mults[1], mults[2] );
        printf("Median calc time us: %d\n", t3.etime(t2));
    }

    t3.set();

    if (ri->getSensorType() == ST_BAYER) {
#ifdef __SSE2__
        const vfloat onev = F2V(1.f);
        const vfloat c65535v = F2V(65535.f);
#endif

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 16)
#endif
        for (int row = 0; row < H; ++row) {
            int col = 0;
            // Exponents are expressed as positive in the parameters, so negate them in order
            // to get the reciprocals. Avoid trouble with zeroes, minimum pixel value is 1.
            const float exps0 = -exps[FC(row, col)];
            const float exps1 = -exps[FC(row, col + 1)];
            const float mult0 = mults[FC(row, col)];
            const float mult1 = mults[FC(row, col + 1)];
#ifdef __SSE2__
            const vfloat expsv = _mm_setr_ps(exps0, exps1, exps0, exps1);
            const vfloat multsv = _mm_setr_ps(mult0, mult1, mult0, mult1);
            for (; col < W - 3; col += 4) {
                STVFU(rawData[row][col], vminf(multsv * pow_F(vmaxf(LVFU(rawData[row][col]), onev), expsv), c65535v));
            }
#endif // __SSE2__
            for (; col < W - 1; col += 2) {
                rawData[row][col] = rtengine::min(mult0 * pow_F(rtengine::max(rawData[row][col], 1.f), exps0), 65535.f);
                rawData[row][col + 1] = rtengine::min(mult1 * pow_F(rtengine::max(rawData[row][col + 1], 1.f), exps1), 65535.f);
            }
            if (col < W) {
                rawData[row][col] = rtengine::min(mult0 * pow_F(rtengine::max(rawData[row][col], 1.f), exps0), 65535.f);
            }
        }
    } else if (ri->getSensorType() == ST_FUJI_XTRANS) {
#ifdef __SSE2__
        const vfloat onev = F2V(1.f);
        const vfloat c65535v = F2V(65535.f);
#endif

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 16)
#endif
        for (int row = 0; row < H; row ++) {
            int col = 0;
            // Exponents are expressed as positive in the parameters, so negate them in order
            // to get the reciprocals. Avoid trouble with zeroes, minimum pixel value is 1.
            const std::array<float, 6> expsc = {
                -exps[ri->XTRANSFC(row, 0)],
                -exps[ri->XTRANSFC(row, 1)],
                -exps[ri->XTRANSFC(row, 2)],
                -exps[ri->XTRANSFC(row, 3)],
                -exps[ri->XTRANSFC(row, 4)],
                -exps[ri->XTRANSFC(row, 5)]
            };
            const std::array<float, 6> multsc = {
                mults[ri->XTRANSFC(row, 0)],
                mults[ri->XTRANSFC(row, 1)],
                mults[ri->XTRANSFC(row, 2)],
                mults[ri->XTRANSFC(row, 3)],
                mults[ri->XTRANSFC(row, 4)],
                mults[ri->XTRANSFC(row, 5)]
            };
#ifdef __SSE2__
            const vfloat expsv0 = _mm_setr_ps(expsc[0], expsc[1], expsc[2], expsc[3]);
            const vfloat expsv1 = _mm_setr_ps(expsc[4], expsc[5], expsc[0], expsc[1]);
            const vfloat expsv2 = _mm_setr_ps(expsc[2], expsc[3], expsc[4], expsc[5]);
            const vfloat multsv0 = _mm_setr_ps(multsc[0], multsc[1], multsc[2], multsc[3]);
            const vfloat multsv1 = _mm_setr_ps(multsc[4], multsc[5], multsc[0], multsc[1]);
            const vfloat multsv2 = _mm_setr_ps(multsc[2], multsc[3], multsc[4], multsc[5]);
            for (; col < W - 11; col += 12) {
                STVFU(rawData[row][col], vminf(multsv0 * pow_F(vmaxf(LVFU(rawData[row][col]), onev), expsv0), c65535v));
                STVFU(rawData[row][col + 4], vminf(multsv1 * pow_F(vmaxf(LVFU(rawData[row][col + 4]), onev), expsv1), c65535v));
                STVFU(rawData[row][col + 8], vminf(multsv2 * pow_F(vmaxf(LVFU(rawData[row][col + 8]), onev), expsv2), c65535v));
            }
#endif // __SSE2__
            for (; col < W - 5; col += 6) {
                for (int c = 0; c < 6; ++c) {
                    rawData[row][col + c] = rtengine::min(multsc[c] * pow_F(rtengine::max(rawData[row][col + c], 1.f), expsc[c]), 65535.f);
                }
            }
            for (int c = 0; col < W; col++, c++) {
                rawData[row][col + c] = rtengine::min(multsc[c] * pow_F(rtengine::max(rawData[row][col + c], 1.f), expsc[c]), 65535.f);
            }
        }
    }

    t4.set();

    if (settings->verbose) {
        printf("Pow loop time us: %d\n", t4.etime(t3));
    }

    t4.set();

    PixelsMap bitmapBads(W, H);

    int totBP = 0; // Hold count of bad pixels to correct

    if (ri->getSensorType() == ST_BAYER) {
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:totBP) schedule(dynamic,16)
#endif
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                if (rawData[i][j] >= MAX_OUT_VALUE) {
                    bitmapBads.set(j, i);
                    ++totBP;
                }
            }
        }

        if (totBP > 0) {
            interpolateBadPixelsBayer(bitmapBads, rawData);
        }

    }
    else if (ri->getSensorType() == ST_FUJI_XTRANS) {
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:totBP) schedule(dynamic,16)
#endif
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                if (rawData[i][j] >= MAX_OUT_VALUE) {
                    bitmapBads.set(j, i);
                    totBP++;
                }
            }
        }

        if (totBP > 0) {
            interpolateBadPixelsXtrans(bitmapBads);
        }
    }

    t5.set();

    if (settings->verbose) {
        printf("Bad pixels count: %d\n", totBP);
        printf("Bad pixels interpolation time us: %d\n", t5.etime(t4));
    }
}
