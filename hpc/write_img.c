#include "write_img.h"

#include <stdio.h>

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
