# Cybervision

![Build status](https://github.com/zlogic/cybervision/actions/workflows/cmake-build.yml/badge.svg)

Cybervision is a 3D reconstruction software for Scanning Electron Microscope images.

The tool needs two images of an object taken from slighly different angles.
Cybervision can match those images and use the parallax effect to determine the object's 3D shape.

⚠️ Only high-contrast images with parallel projection are supported.
Regular photos are not likely to work correctly, because regular cameras have perspective projection.

More information is available in the [Wiki](https://github.com/zlogic/cybervision/wiki).

## How to use it

Download a release distribution from [releases](/zlogic/cybervision/releases).

Run cybervision:

```shell
cybervision [--scale=<scale>] [--mode=<cpu|gpu>] [--interpolation=<none|delaunay>] [--projection=<parallel|perspective>] <img1> <img2> <output>
```

`--scale=<scale>` is an optional argument to specify a depth scale, for example `--scale=-10.0`.

`--mode=<cpu|gpu>` is an optional argument to specify a depth scale, for example `--mode=cpu` or `--mode=gpu`.Results might be slightly different between modes because the implementation is not completely identical.

`--interpolation=<none|delaunay>` is an optional argument to specify a depth scale, for example `--mode=cpu` or `--mode=gpu`. 
`none` means that interpolation is disabled, `delaunay` uses [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation).

`--projection=<parallel|perspective>` is an optional argument to specify a projection mode, for example `--projection=parallel` or `--projection=perspective`. 
`parallel` projection should be used for images from a scanning electron microscope, `perspective` should be used for regular photos.

`<img1>` and `<img2>` are input filenames for image 1 and 2; supported formats are `jpg`, `tif` and `png`.

`<output>` is the output filename:
* If the filename ends with `.obj`, this will save a 3D [Wavefront OBJ file](https://en.wikipedia.org/wiki/Wavefront_.obj_file).
* If the filename ends with `.ply`, this will save a 3D [PLY binary file](https://en.wikipedia.org/wiki/PLY_(file_format)).
* If the filename ends with `.png`, this will save a PNG depth map file.

⚠️ The optimal image size is 1024x1024 (or similar).
Using larger images will result in increased processing times, increased memory usage, and run into GPU hardware limitations.
Smaller images might not have enough details.

### GPU details

Compiling a GPU-accelerated (Vulkan) version requires additional libraries and build tools.

Cybervision was tested to support CPU-only and GPU-accelerated processing on:

* Apple Macbook Air M1 (2020)
* Apple Macbook Pro M1 Max (2021)
* Windows 11, i7-8750H, Geforce GTX 1050 (mobile)
* Fedora CoreOS 36, Celeron N3350 (digital signage appliance)

To run Cybervision, you will need the Vulkan runtime library:

* In Linux, it's called `libvulkan.so.1` and the package is typically called something like `vulkan`, `vulkan-loader`, `libvulkan` or `libvulkan1`.
* In Windows, it's the Vulkan Runtime (VulkanRT) should already be installed - it's included with GPU drivers.
* In macOS, a native Metal implementation is used instead of Vulkan, no extra libraries are needed.

More details can be found in [gpu.md](gpu.md).

## C version

Cybervision was rewritten in C.

Originally, it was a full all-in-one tool built based on Qt and using a different approach.
For more details about the C++ version, see [Releases](/zlogic/cybervision/releases).
The source code is available in the [branch_qt_sift](../../tree/branch_qt_sift) branch.

The C rewrite focuses on the primary goal - generating a 3D surface from an image stereopair;
anything else (like a UI) can be added separately.

* Using a much simpler model, while collecting a lot more detail
* Removed potential liabilities such as
  * Patented algorithms (SIFT, although it's likely to expire soon)
  * Unmaintained libraries (siftfast)
  * LGPL licensed code (siftfast, Qt)
  * Legacy or non-portable APIs and standards (OpenCL, OpenMP, SSE)
  * C++ template code that crashed Qt Creator's clang

## How to build it

Cybervision uses CMake and should work on Linux, macOS and Windows.

To build it:

1. Install [vcpkg](https://vcpkg.io/en/index.html) - used to download and build all external dependencies.
2. Install Vulkan SDK (Windows and Linux only), see [gpu.md](gpu.md) for mode details.
3. Configure and build the project with CMake.

## External libraries or dependencies

* [fast](https://www.edwardrosten.com/work/fast.html) keypoint detector
* [qhull](http://www.qhull.org) for Delaunay triangulation
* [libtiff](http://libtiff.maptools.org/) to load `tiff` files
* [libjpeg-turbo](https://libjpeg-turbo.org) to load `jpeg` files
* [libpng](http://libpng.org/pub/png/libpng.html) to load and save `png` files
* [plasma colormap](https://bids.github.io/colormap/) to generate depth map images
