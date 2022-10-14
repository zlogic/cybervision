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
        png_structp png_ptr;
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
        png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
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
        for (size_t i=0;i<img->height;i++)
        {
            row_pointers[i] = (png_byte*)malloc(png_get_rowbytes(png_ptr, info_ptr));
        }
        png_read_image(png_ptr, row_pointers);

        for(size_t y=0;y<img->height;y++)
        {
            png_byte* row = row_pointers[y];
            for (size_t x=0;x<img->width;x++)
            {
                png_byte pixel = row[x];
                img->img[y*img->width+x] = pixel;
            }
        }
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        for (size_t i=0;i<img->height;i++)
        {
            free(row_pointers[i]);
        }
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

static inline png_byte map_color(float value, const png_byte *colormap, size_t colormap_size)
{
    if (value <= 0)
        return colormap[0];
    if (value >= 1.0F)
        return colormap[colormap_size-1];
    float step = 1.0F/(colormap_size-1);
    size_t box = (size_t)floorf(value/step);
    float ratio = (value-step*box)/step;
    png_byte c1 = colormap[box];
    png_byte c2 = (box+1)<colormap_size? colormap[box+1]:colormap[box];
    int color = (int)roundf((float)c2*ratio+(float)c1*(1.0F-ratio));
    if (color>0xFF)
        return 0xFF;
    if (color<0)
        return 0;
    return (png_byte)color;
}

int save_surface_image(surface_data data, char *filename)
{
    FILE *pngFile;
    png_structp png_ptr;
    png_infop info_ptr;
    png_bytep *row_pointers;

    // viridis from https://bids.github.io/colormap/
    static const png_byte colormap_r[] = {0xfd, 0xfb, 0xf8, 0xf6, 0xf4, 0xf1, 0xef, 0xec, 0xea, 0xe7, 0xe5, 0xe2, 0xdf, 0xdd, 0xda, 0xd8, 0xd5, 0xd2, 0xd0, 0xcd, 0xca, 0xc8, 0xc5, 0xc2, 0xc0, 0xbd, 0xba, 0xb8, 0xb5, 0xb2, 0xb0, 0xad, 0xaa, 0xa8, 0xa5, 0xa2, 0xa0, 0x9d, 0x9b, 0x98, 0x95, 0x93, 0x90, 0x8e, 0x8b, 0x89, 0x86, 0x84, 0x81, 0x7f, 0x7c, 0x7a, 0x77, 0x75, 0x73, 0x70, 0x6e, 0x6c, 0x69, 0x67, 0x65, 0x63, 0x60, 0x5e, 0x5c, 0x5a, 0x58, 0x56, 0x54, 0x52, 0x50, 0x4e, 0x4c, 0x4a, 0x48, 0x46, 0x44, 0x42, 0x40, 0x3f, 0x3d, 0x3b, 0x3a, 0x38, 0x37, 0x35, 0x34, 0x32, 0x31, 0x2f, 0x2e, 0x2d, 0x2c, 0x2a, 0x29, 0x28, 0x27, 0x26, 0x25, 0x25, 0x24, 0x23, 0x22, 0x22, 0x21, 0x21, 0x20, 0x20, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1e, 0x1e, 0x1e, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x20, 0x20, 0x20, 0x21, 0x21, 0x21, 0x21, 0x22, 0x22, 0x22, 0x23, 0x23, 0x23, 0x24, 0x24, 0x25, 0x25, 0x25, 0x26, 0x26, 0x26, 0x27, 0x27, 0x27, 0x28, 0x28, 0x29, 0x29, 0x29, 0x2a, 0x2a, 0x2a, 0x2b, 0x2b, 0x2c, 0x2c, 0x2c, 0x2d, 0x2d, 0x2e, 0x2e, 0x2e, 0x2f, 0x2f, 0x30, 0x30, 0x31, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33, 0x34, 0x34, 0x35, 0x35, 0x36, 0x36, 0x37, 0x37, 0x38, 0x38, 0x39, 0x39, 0x3a, 0x3a, 0x3b, 0x3b, 0x3c, 0x3c, 0x3d, 0x3d, 0x3e, 0x3e, 0x3e, 0x3f, 0x3f, 0x40, 0x40, 0x41, 0x41, 0x42, 0x42, 0x42, 0x43, 0x43, 0x44, 0x44, 0x44, 0x45, 0x45, 0x45, 0x46, 0x46, 0x46, 0x46, 0x47, 0x47, 0x47, 0x47, 0x47, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x47, 0x47, 0x47, 0x47, 0x47, 0x46, 0x46, 0x46, 0x46, 0x45, 0x45, 0x44, 0x44};
    static const png_byte colormap_g[] = {0xe7, 0xe7, 0xe6, 0xe6, 0xe6, 0xe5, 0xe5, 0xe5, 0xe5, 0xe4, 0xe4, 0xe4, 0xe3, 0xe3, 0xe3, 0xe2, 0xe2, 0xe2, 0xe1, 0xe1, 0xe1, 0xe0, 0xe0, 0xdf, 0xdf, 0xdf, 0xde, 0xde, 0xde, 0xdd, 0xdd, 0xdc, 0xdc, 0xdb, 0xdb, 0xda, 0xda, 0xd9, 0xd9, 0xd8, 0xd8, 0xd7, 0xd7, 0xd6, 0xd6, 0xd5, 0xd5, 0xd4, 0xd3, 0xd3, 0xd2, 0xd1, 0xd1, 0xd0, 0xd0, 0xcf, 0xce, 0xcd, 0xcd, 0xcc, 0xcb, 0xcb, 0xca, 0xc9, 0xc8, 0xc8, 0xc7, 0xc6, 0xc5, 0xc5, 0xc4, 0xc3, 0xc2, 0xc1, 0xc1, 0xc0, 0xbf, 0xbe, 0xbd, 0xbc, 0xbc, 0xbb, 0xba, 0xb9, 0xb8, 0xb7, 0xb6, 0xb6, 0xb5, 0xb4, 0xb3, 0xb2, 0xb1, 0xb0, 0xaf, 0xae, 0xad, 0xad, 0xac, 0xab, 0xaa, 0xa9, 0xa8, 0xa7, 0xa6, 0xa5, 0xa4, 0xa3, 0xa2, 0xa1, 0xa1, 0xa0, 0x9f, 0x9e, 0x9d, 0x9c, 0x9b, 0x9a, 0x99, 0x98, 0x97, 0x96, 0x95, 0x94, 0x93, 0x92, 0x92, 0x91, 0x90, 0x8f, 0x8e, 0x8d, 0x8c, 0x8b, 0x8a, 0x89, 0x88, 0x87, 0x86, 0x85, 0x84, 0x83, 0x82, 0x82, 0x81, 0x80, 0x7f, 0x7e, 0x7d, 0x7c, 0x7b, 0x7a, 0x79, 0x78, 0x77, 0x76, 0x75, 0x74, 0x73, 0x72, 0x71, 0x71, 0x70, 0x6f, 0x6e, 0x6d, 0x6c, 0x6b, 0x6a, 0x69, 0x68, 0x67, 0x66, 0x65, 0x64, 0x63, 0x62, 0x61, 0x60, 0x5f, 0x5e, 0x5d, 0x5c, 0x5b, 0x5a, 0x59, 0x58, 0x56, 0x55, 0x54, 0x53, 0x52, 0x51, 0x50, 0x4f, 0x4e, 0x4d, 0x4c, 0x4a, 0x49, 0x48, 0x47, 0x46, 0x45, 0x44, 0x42, 0x41, 0x40, 0x3f, 0x3e, 0x3d, 0x3b, 0x3a, 0x39, 0x38, 0x37, 0x35, 0x34, 0x33, 0x32, 0x30, 0x2f, 0x2e, 0x2d, 0x2c, 0x2a, 0x29, 0x28, 0x26, 0x25, 0x24, 0x23, 0x21, 0x20, 0x1f, 0x1d, 0x1c, 0x1b, 0x1a, 0x18, 0x17, 0x16, 0x14, 0x13, 0x11, 0x10, 0x0e, 0x0d, 0x0b, 0x0a, 0x08, 0x07, 0x05, 0x04, 0x02, 0x01};
    static const png_byte colormap_b[] = {0x25, 0x23, 0x21, 0x20, 0x1e, 0x1d, 0x1c, 0x1b, 0x1a, 0x19, 0x19, 0x18, 0x18, 0x18, 0x19, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1f, 0x20, 0x21, 0x23, 0x25, 0x26, 0x28, 0x29, 0x2b, 0x2d, 0x2f, 0x30, 0x32, 0x34, 0x36, 0x37, 0x39, 0x3b, 0x3c, 0x3e, 0x40, 0x41, 0x43, 0x45, 0x46, 0x48, 0x49, 0x4b, 0x4d, 0x4e, 0x50, 0x51, 0x53, 0x54, 0x56, 0x57, 0x58, 0x5a, 0x5b, 0x5c, 0x5e, 0x5f, 0x60, 0x62, 0x63, 0x64, 0x65, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x79, 0x7a, 0x7b, 0x7c, 0x7c, 0x7d, 0x7e, 0x7f, 0x7f, 0x80, 0x81, 0x81, 0x82, 0x82, 0x83, 0x83, 0x84, 0x85, 0x85, 0x85, 0x86, 0x86, 0x87, 0x87, 0x88, 0x88, 0x88, 0x89, 0x89, 0x89, 0x8a, 0x8a, 0x8a, 0x8b, 0x8b, 0x8b, 0x8b, 0x8c, 0x8c, 0x8c, 0x8c, 0x8c, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d, 0x8c, 0x8c, 0x8c, 0x8c, 0x8c, 0x8c, 0x8b, 0x8b, 0x8b, 0x8b, 0x8a, 0x8a, 0x8a, 0x8a, 0x89, 0x89, 0x89, 0x88, 0x88, 0x88, 0x87, 0x87, 0x86, 0x86, 0x85, 0x85, 0x84, 0x84, 0x83, 0x83, 0x82, 0x81, 0x81, 0x80, 0x7f, 0x7e, 0x7e, 0x7d, 0x7c, 0x7b, 0x7a, 0x7a, 0x79, 0x78, 0x77, 0x76, 0x75, 0x74, 0x73, 0x71, 0x70, 0x6f, 0x6e, 0x6d, 0x6c, 0x6a, 0x69, 0x68, 0x67, 0x65, 0x64, 0x63, 0x61, 0x60, 0x5e, 0x5d, 0x5c, 0x5a, 0x59, 0x57, 0x56, 0x54};
    static const size_t colormap_size = 256;

    float min_depth = INFINITY, max_depth = -INFINITY;
    int w, h;
    if ((pngFile = fopen(filename, "wb")) == NULL)
        return 0;
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
    {
        fclose(pngFile);
        return 0;
    }
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        fclose(pngFile);
        return 0;
    }
    png_init_io(png_ptr, pngFile);
    png_set_IHDR(png_ptr, info_ptr, data.width, data.height, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);

    row_pointers = malloc(sizeof(png_bytep)*data.height);
    for (size_t i=0;i<data.height;i++)
    {
        row_pointers[i] = (png_byte*)malloc(png_get_rowbytes(png_ptr, info_ptr));
    }

    for(size_t i=0;i<data.width*data.height;i++)
    {
        float depth = data.depth[i];
        min_depth = depth<min_depth? depth:min_depth;
        max_depth = depth>max_depth? depth:max_depth;
    }

    for(size_t y=0;y<data.height;y++)
    {
        png_byte* row = row_pointers[y];
        for (size_t x=0;x<data.width;x++)
        {
            float depth = (data.depth[y*data.width+x]-min_depth)/(max_depth-min_depth);
            if (!isfinite(depth))
            {
                row[x*4] = 0; row[x*4+1] = 0; row[x*4+2] = 0;
                row[x*4+3] = 0;
            }
            else
            {
                row[x*4] = map_color(depth, colormap_r, colormap_size);
                row[x*4+1] = map_color(depth, colormap_g, colormap_size);
                row[x*4+2] = map_color(depth, colormap_b, colormap_size);
                row[x*4+3] = 0xFF;
            }
        }
    }
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    for (size_t i=0;i<data.height;i++)
    {
        free(row_pointers[i]);
    }
    free(row_pointers);
    fclose(pngFile);
    return 1;
}
