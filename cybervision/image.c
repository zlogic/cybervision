#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <tiffio.h>
#include <jpeglib.h>
#include <png.h>

#include "system.h"
#include "image.h"

/*
 * FEI tags containing image details
 */
#define TIFFTAG_META_PHENOM 34683
#define TIFFTAG_META_QUANTA 34682

#define PNG_BYTES_TO_CHECK 8

char *file_extension(char *filename)
{
    char* dot_i = strrchr(filename, '.');
    if (dot_i == NULL || dot_i == filename && strlen(dot_i) == 0)
        return "";
    return dot_i+1;
}

static inline char convert_grayscale(int r, int g, int b)
{
    return (char)((19595*r + 38470*g + 7471*b)>>16);
}

int read_tiff_tags(TIFF* tif, int* databar_height)
{
    void* tag_data = NULL;
    uint32_t tag_count;
    char *line = NULL, *line_mem = NULL;
    char *section = NULL;

    if (!TIFFGetField(tif, TIFFTAG_META_PHENOM, &tag_count, &tag_data) && !TIFFGetField(tif, TIFFTAG_META_QUANTA, &tag_count, &tag_data))
        return 0;
    
    line = malloc(sizeof(char)*(tag_count+1));
    memcpy(line, tag_data, tag_count);
    line_mem = line;
    line[tag_count] = '\0';
    while (line != NULL && *line != '\0')
    {
        char *newline = strchr(line, '\n');
        size_t line_len = newline==NULL? strlen(line):newline-line;
        char *cr = strchr(line, '\r');
        if (cr != NULL && cr-line < line_len)
            line_len = cr-line;
        line[line_len] = '\0';
        if (line[0]== '[')
            section = line;
        /*
        // TODO: parse and use scale
        if (strcmp(section, "[Scan]") == 0)
        {
            if (strncmp(line, "PixelWidth=", 11) == 0)
            {
                float pixel_width = strtof(line+11, NULL);
                printf("width=%f\n", pixel_width);
            }
            else if (strncmp(line, "PixelHeight=", 12) == 0)
            {
                float pixel_height = strtof(line+12, NULL);
                printf("height=%f\n", pixel_height);
            }
        }
        // TODO: use rotation (see "Real scale (Tomasi) stuff.pdf")
        // or allow to specify a custom depth scale (e.g. a negative one)
        if (strcmp(section, "[Stage]") == 0)
        {
            if (strncmp(line, "StageT=", 7) == 0)
            {
                float rotation = strtof(line+7, NULL);
                printf("rotation=%f\n", rotation);
            }
        }
        */
        if (strcmp(section, "[PrivateFei]") == 0)
            if (strncmp(line, "DatabarHeight=", 14) == 0)
                *databar_height = atoi(line+14);
        line += line_len+1;
    }

    free(line_mem);
    return 0;
}

correlation_image* load_image(char *filename)
{
    char *file_ext = file_extension(filename);
    correlation_image *img = NULL;
    if (strcasecmp(file_ext, "tif") == 0 || strcasecmp(file_ext, "tiff") == 0)
    {
        TIFFSetWarningHandler(NULL);
        TIFF* tif = TIFFOpen(filename, "r");
        if (!tif)
            return NULL;
        uint32_t w, h;
        int databar_height = 0;

        size_t npixels;
        uint32_t* raster;

        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
        
        read_tiff_tags(tif, &databar_height);

        npixels = w * h;
        raster = (uint32_t*) _TIFFmalloc(npixels * sizeof (uint32_t));
        if (raster != NULL)
        {
            if (TIFFReadRGBAImage(tif, w, h, raster, 0))
            {
                img = malloc(sizeof(correlation_image));
                img->width = w;
                img->height = h-databar_height;
                img->img = malloc(sizeof(unsigned)*img->width*img->height);
                for (size_t y=0;y<img->height;y++)
                {
                    for (size_t x=0;x<img->width;x++)
                    {
                        uint32_t pixel = raster[(h-y-1)*w + x];
                        img->img[y*w+x] = convert_grayscale(TIFFGetR(pixel), TIFFGetB(pixel), TIFFGetB(pixel));
                    }
                }
            }
            _TIFFfree(raster);
        }
        TIFFClose(tif);
    }
    else if (strcasecmp(file_ext, "jpg") == 0 || strcasecmp(file_ext, "jpeg") == 0) 
    {
        FILE *jpegFile = NULL;
        int row_stride;
        JSAMPARRAY buffer;
        struct jpeg_decompress_struct cinfo;
        struct jpeg_error_mgr jerr;
        
        if ((jpegFile = fopen(filename, "rb")) == NULL)
            return NULL;

        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_decompress(&cinfo);
        cinfo.out_color_space = JCS_RGB;
        jpeg_stdio_src(&cinfo, jpegFile);
        (void)jpeg_read_header(&cinfo, TRUE);
        (void)jpeg_start_decompress(&cinfo);
        row_stride = cinfo.output_width * cinfo.output_components;
        buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

        img = malloc(sizeof(correlation_image));
        img->width = cinfo.output_width;
        img->height = cinfo.output_height;
        img->img = malloc(sizeof(unsigned)*img->width*img->height);

        while (cinfo.output_scanline < cinfo.output_height) {
            int y = cinfo.output_scanline;
            (void)jpeg_read_scanlines(&cinfo, buffer, 1);
            for (int i=0;i<img->width;i++)
            {
                JSAMPLE *pixel = &buffer[0][i*cinfo.output_components];
                img->img[y*img->width+i] = convert_grayscale(*pixel, *(pixel+1), *(pixel+2));
            }
        }
        (void)jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(jpegFile);
    }
    else if (strcasecmp(file_ext, "png") == 0) 
    {
        FILE *pngFile;
        png_byte header[PNG_BYTES_TO_CHECK];
        png_infop info_ptr;
        png_bytep *row_pointers;
        png_byte color_type;
        int w, h;
        if ((pngFile = fopen(filename, "rb")) == NULL)
            return NULL;
        if (fread(header, 1, PNG_BYTES_TO_CHECK, pngFile) != PNG_BYTES_TO_CHECK)
        {
            fclose(pngFile);
            return NULL;
        }
        if (png_sig_cmp(header, 0, PNG_BYTES_TO_CHECK) != 0)
        {
            fclose(pngFile);
            return NULL;
        }
        fseek(pngFile, 0, SEEK_SET);
        png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        info_ptr = png_create_info_struct(png_ptr);
        png_init_io(png_ptr, pngFile);
        png_read_info(png_ptr, info_ptr);
        color_type = png_get_color_type(png_ptr, info_ptr);
        if (color_type & PNG_COLOR_TYPE_GRAY_ALPHA)
            png_set_strip_alpha(png_ptr);
        if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_RGB_ALPHA)
            png_set_rgb_to_gray_fixed(png_ptr, PNG_ERROR_ACTION_NONE, PNG_RGB_TO_GRAY_DEFAULT, PNG_RGB_TO_GRAY_DEFAULT);
        png_read_update_info(png_ptr, info_ptr);

        img = malloc(sizeof(correlation_image));
        img->width = png_get_image_width(png_ptr, info_ptr);
        img->height = png_get_image_height(png_ptr, info_ptr);
        img->img = malloc(sizeof(unsigned)*img->width*img->height);

        row_pointers = malloc(sizeof(png_bytep)*img->height);
        for(size_t i=0;i<img->height;i++) {
            row_pointers[i] = (png_byte*)malloc(png_get_rowbytes(png_ptr, info_ptr));
        }
        png_read_image(png_ptr, row_pointers);

        for(size_t y=0;y<img->height;y++) {
            png_byte* row = row_pointers[y];
            for (size_t x=0;x<img->width;x++)
            {
                png_byte pixel = row[x];
                img->img[y*img->width+x] = pixel;
            }
        }
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL); 
        free(row_pointers);
        fclose(pngFile);
    }
    return img;
}

void resize_image(correlation_image *src, correlation_image *dst, float scale)
{
    dst->width = (int)floorf(src->width*scale);
    dst->height = (int)floorf(src->height*scale);
    dst->img = malloc(sizeof(unsigned char)*dst->width*dst->height);
    for (int y=0;y<dst->height;y++)
    {
        int y0 = (int)floor(y/scale);
        int y1 = (int)ceil((y+1)/scale);
        y1 = y1<src->height? y1 : src->height-1;
        for (int x=0;x<dst->width;x++)
        {
            int x0 = (int)roundf((float)x/scale);
            int x1 = (int)ceil((x+1)/scale);
            x1 = x1<src->width? x1 : src->width-1;
            float value = 0.0F;
            float coeffs = 0.0F;
            for (int j=y0;j<y1;j++)
            {
                float y_coeff = 1.0F-(j-y0)*scale;
                for (int i=x0;i<x1;i++)
                {
                    float x_coeff = 1.0F-(i-x0)*scale;
                    value += y_coeff*x_coeff*src->img[j*src->width + i];
                    coeffs += y_coeff*x_coeff;
                }
            }
            dst->img[y*dst->width + x] = (int)roundf(value/coeffs);
        }
    }
}
