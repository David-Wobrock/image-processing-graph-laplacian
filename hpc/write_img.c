#include "write_img.h"

#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "utils.h"

/*
vec_mat is a 1xN matrix with values between 0 and 1
Returns a filled png_bytes for proc with rank 0 only
Returns NULL for the others
*/
png_bytep* VecMat2pngbytes(Mat vec_mat, const unsigned int width, const unsigned int height, const int scale)
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // Allocate memory for image for proc 0
    png_bytep* img_bytes = NULL;
    if (rank == 0)
    {
        img_bytes = (png_bytep*) malloc(sizeof(png_bytep) * height);
        for (unsigned int i = 0; i < height; ++i)
        {
            img_bytes[i] = (png_bytep) malloc(sizeof(png_byte) * width);
        }

        // All data is on process 0 already, get all values and cast to png_byte
        PetscInt* col_indices = (PetscInt*) malloc(sizeof(PetscInt) * width);
        PetscScalar* values = (PetscScalar*) malloc(sizeof(PetscScalar) * width);
        for (unsigned int i = 0; i < height; ++i)
        {
            for (unsigned int j = 0; j < width; ++j)
            {
                col_indices[j] = j + width*i;
            }
            MatGetValues(vec_mat, 1, &ZERO, width, col_indices, values);
            for (unsigned int j = 0; j < width; ++j)
            {
                img_bytes[i][j] = ((png_byte) (values[j] * scale));
            }
        }
        free(values);
        free(col_indices);
    }

    // Gather all data to process 0
    return img_bytes;
}

int write_png(const char* const filename, png_bytep* img_bytes, const unsigned int width, const unsigned int height)
{
    FILE* f_ptr = fopen(filename, "wb");
    if (!f_ptr)
    {
        fprintf(stderr, "Could not open file %s\n", filename);
        return -1;
    }

    png_structp png_ptr = png_create_write_struct(
        PNG_LIBPNG_VER_STRING,
        NULL, NULL, NULL);
    if (!png_ptr)
    {
        return -1;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_write_struct(&png_ptr, NULL);
        return -1;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(f_ptr);
        return -1;
    }

    png_init_io(png_ptr, f_ptr);

    png_byte color_type = PNG_COLOR_TYPE_GRAY;
    png_byte bit_depth = 8;
    // Write header
    png_set_IHDR(
        png_ptr, info_ptr, width, height,
        bit_depth, color_type, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png_ptr, info_ptr);

    // Write content
    png_write_image(png_ptr, img_bytes);
    png_write_end(png_ptr, NULL);

    fclose(f_ptr);
    return 0;
}
