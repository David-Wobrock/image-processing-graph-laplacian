#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <png.h>

#include "utils.h"

int read_png(const char* const filename, png_bytep** row_pointers, int* const width, int* const height)
{
    png_byte color_type;

    FILE* f_ptr = fopen(filename, "rb");
    if (!f_ptr)
    {
        fprintf(stderr, "Could not open file %s\n", filename);
        return -1;
    }

    png_structp png_ptr = png_create_read_struct(
        PNG_LIBPNG_VER_STRING,
        NULL, NULL, NULL);
    if (!png_ptr) return -1;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return -1;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(f_ptr);
        return -1;
    }

    png_init_io(png_ptr, f_ptr);

    png_read_info(png_ptr, info_ptr);

    *width = png_get_image_width(png_ptr, info_ptr);
    *height = png_get_image_height(png_ptr, info_ptr);
    color_type = png_get_color_type(png_ptr, info_ptr);

    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_RGB_ALPHA)
    {
        png_set_rgb_to_gray(png_ptr, 1, -1, -1);
    }

    png_read_update_info(png_ptr, info_ptr);

    unsigned int num_row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    *row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * *height);
    for (unsigned int i = 0; i < *height; ++i)
    {
        (*row_pointers)[i] = (png_bytep) malloc(num_row_bytes);
    }

    png_read_image(png_ptr, *row_pointers);

    fclose(f_ptr);
    return 0;
}
