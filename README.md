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
cybervision<img1.tif> <img2.tif> <out.obj>
```

This will save a 3D [Wavefront OBJ file](https://en.wikipedia.org/wiki/Wavefront_.obj_file).

⚠️ The ideal image size is 1024x1024 (or similar). Using larger images might result in increased processing times. Smaller images might not have enough details.

### GPU details

Previous versions of Cybervision were relying on a GPU as a way to brute force and correlate as many points as possible.

Thanks to optimizations, this is no longer necessary and the CPU-based version runs well enough.
Removing Vulkan and Metal makes Cybervision a lot more portable.

For more details how the GPU version used to work, see the [tag_python_gpu](../../tree/tag_python_gpu) tag.

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

## External libraries

* [fast](https://www.edwardrosten.com/work/fast.html) keypoint detector
* [qhull](http://www.qhull.org) for Delaunay triangulation
* [libtiff](http://libtiff.maptools.org/) to load `tiff` files
* [libjpeg-turbo](https://libjpeg-turbo.org) to load `jpeg` files
