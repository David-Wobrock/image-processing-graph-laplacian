#include "sampling.h"

#include <stdlib.h>
#include <math.h>

static void UniformSampling(const int width, const int height, unsigned int* const sample_size, unsigned int** const sample_indices)
{
    const unsigned int sample_dist = (unsigned int) (sqrt((width*height) / (*sample_size)));
    const unsigned int xy0 = (unsigned int) (sample_dist/2);
    const unsigned int size_x_span = (unsigned int) ceil((height - 1 - xy0) / (double) sample_dist);
    const unsigned int size_y_span = (unsigned int) ceil((width - 1 - xy0) / (double) sample_dist);

    *sample_size = size_x_span * size_y_span;
    *sample_indices = (unsigned int*) malloc(sizeof(unsigned int) * (*sample_size));
    unsigned int c = 0;
    for (unsigned int i = xy0; i < height-1; i += sample_dist)
    {
        for (unsigned int j = xy0; j < width-1; j += sample_dist)
        {
            (*sample_indices)[c++] = width*i + j;
        }
    }
}

/*
Get indices of the sampled pixels (from 0 to width*height)
Input: image width and height, the number of requested samples (as pointer because it will be modified), a *not* allocated pointer for the indices
Output: the exact number of samples (sample_size) and a filled and allocated array with the indices (sample_indices)
*/
void Sampling(const int width, const int height, unsigned int* const sample_size, unsigned int** const sample_indices)
{
    UniformSampling(width, height, sample_size, sample_indices);
}
