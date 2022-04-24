# Cybervision

![Build status](https://github.com/zlogic/cybervision/actions/workflows/python-package.yml/badge.svg)

Cybervision is a 3D reconstruction software for Scanning Electron Microscope images.

The tool needs two images of an object taken from slighly different angles.
Cybervision can match those images and use the parallax effect to determine the object's 3D shape.

⚠️ Only high-contrast images with parallel projection are supported.
Regular photos are not likely to work correctly, because regular cameras have perspective projection.

More information is available in the [Wiki](/zlogic/cybervision/wiki).

## How to use it

Download a release .whl file for your platform from [releases](/zlogic/cybervision/releases).

Install the .whl file by running:

```sheell
pip3 install <filename>.whl
```

Run cybervision:

```shell
python3 -m cybervision <img1.tif> <img2.tif> --output-file=<out.png> [--no-interpolate]
```

## Python version

Cybervision was rewritten in Python (with C extensions).

Originally, it was a full all-in-one tool built based on Qt and using a different approach.
For more details about the C++ version, see [Releases](/zlogic/cybervision/releases).
The source code is available in the [branch_qt_sift](/zlogic/cybervision/tree/branch_qt_sift) branch.

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
* [scipy](https://scipy.org) and [numpy](https://numpy.org) for point interpolation
* [Pillow](https://python-pillow.org) image library
