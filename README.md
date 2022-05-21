# Cybervision

![Build status](https://github.com/zlogic/cybervision/actions/workflows/python-package.yml/badge.svg)

Cybervision is a 3D reconstruction software for Scanning Electron Microscope images.

The tool needs two images of an object taken from slighly different angles.
Cybervision can match those images and use the parallax effect to determine the object's 3D shape.

⚠️ Only high-contrast images with parallel projection are supported.
Regular photos are not likely to work correctly, because regular cameras have perspective projection.

More information is available in the [Wiki](https://github.com/zlogic/cybervision/wiki).

## How to use it

Download a release .whl file for your platform from [releases](/zlogic/cybervision/releases).

Install the .whl file by running:

```sheell
pip3 install <filename>.whl
```

Run cybervision:

```shell
python3 -m cybervision <img1.tif> <img2.tif> --output-file=<out.obj> [--no-interpolate]
```

This will save a 3D [Wavefront OBJ file](https://en.wikipedia.org/wiki/Wavefront_.obj_file).

⚠️ The ideal image size is 1024x1024 (or similar). Using larger images might result in increased processing times, and might cause the GPU to time out ("Device Lost" errors). Smaller images might not have enough details.

### GPU details

Compiling a GPU-accelerated (Vulkan) version requires additional libraries and build tools.

Cybervision was tested on:

* Apple Macbook Air M1 (2020)
* Windows 11, i7-8750H, Geforce GTX 1050 (mobile)

To run Cybervision, you will need the Vulkan runtime library:

* In Linux, it's called `libvulkan.so.1` and the package is typically called something like `vulkan`, `vulkan-loader` or `libvulkan`.
* In Windows, it's the Vulkan Runtime (VulkanRT) should already be installed - it's included with GPU drivers.
* In macOS, a native Metal implementation is used instead of Vulkan.

More details can be found in [gpu.md](gpu.md).

## Python version

Cybervision was rewritten in Python (with C extensions).

Originally, it was a full all-in-one tool built based on Qt and using a different approach.
For more details about the C++ version, see [Releases](/zlogic/cybervision/releases).
The source code is available in the [branch_qt_sift](../../tree/branch_qt_sift) branch.

The Python rewrite focuses on the primary goal - generating a 3D surface from an image stereopair;
anything else (like a UI) can be added separately.

* Using a much simpler model, while collecting a lot more detail
* Removed potential liabilities such as
  * Patented algorithms (SIFT, although it's likely to expire soon)
  * Unmaintained libraries (siftfast)
  * LGPL licensed code (siftfast, Qt)
  * Legacy or non-portable APIs and standards (OpenCL, OpenMP, SSE)
  * C++ template code that crashed Qt Creator's clang

## External libraries

* [fast](https://www.edwardrosten.com/work/fast.html) keypoint detector
* [qhull](http://www.qhull.org) for Delaunay triangulation
* [Pillow](https://python-pillow.org) image library
