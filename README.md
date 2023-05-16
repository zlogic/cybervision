# Cybervision

# Experimental multi view branch

**⚠️ Warning** This `multi-view-reconstruction` is an experimental branch, attempting to reconstruct 3D images from a set of photos.
Unfortunately, it requires either camera calibration, or using [advanced math for self-calibration](https://www.researchgate.net/publication/3659897_The_modulus_constraint_A_new_constraint_self-calibration) - solving a quartic equation system with up to 64 roots.
It's a bit too much and out of my scope of expertise.
Without calibration, each pair of images uses its own coordinate system and it's impossible to align images in a single 3D model.
It also doesn't help that sometimes noise or incorrectly detected features cause images to be incorrectly reconstructed, and adding that data to a combined mesh ruins the end result.

![Build status](https://github.com/zlogic/cybervision/actions/workflows/cargo-build.yml/badge.svg)

<img src="https://raw.githubusercontent.com/wiki/zlogic/cybervision/Cybervision.svg" width="100"/>

Cybervision is a 3D reconstruction software for Scanning Electron Microscope images.

The tool needs two images of an object taken from slighly different angles.
Cybervision can match those images and use the parallax effect to determine the object's 3D shape.

⚠️ Cybervision works best with high-contrast images with parallel (affine) projection.
Regular photos with perspective projection can be reconstructed as well, but this is a secondary use case.

More information is available in the [Wiki](https://github.com/zlogic/cybervision/wiki).

<img src="https://raw.githubusercontent.com/wiki/zlogic/cybervision/Explanation/03_mesh_small.png"/>

<img src="https://raw.githubusercontent.com/wiki/zlogic/cybervision/Examples/Photos/photo4-small.jpg"/>

## How to use it

Download a release distribution from [releases](/zlogic/cybervision/releases).

Run cybervision:

```shell
cybervision [--scale=<scale>] [--mode=<cpu|gpu>] [--interpolation=<none|delaunay>] [--projection=<parallel|perspective>] [--mesh=<plain|vertex-colors|texture-coordinates>] [--no-bundle-adjustment] <img1> <img2> <output>
```

`--scale=<scale>` is an optional argument to specify a depth scale, for example `--scale=-10.0`.

`--mode=<cpu|gpu|gpu-low-power>` is an optional argument to specify a depth scale, for example `--mode=cpu` or `--mode=gpu`
 Results might be slightly different between modes because the implementation is not completely identical.
 `gpu-low-power` will prefer a low-power GPU (like an integrated one) and will reduce the batch size to prevent errors (at the cost of reduced performance).

`--interpolation=<none|delaunay>` is an optional argument to specify an interpolation mode, for example `--interpolation=none` or `--interpolation=delaunay`. 
`none` means that interpolation is disabled, `delaunay` uses [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation).

`--projection=<parallel|perspective>` is an optional argument to specify a projection mode, for example `--projection=parallel` or `--projection=perspective`. 
`parallel` projection should be used for images from a scanning electron microscope, `perspective` should be used for photos from a regular camera.

`--mesh=<plain|vertex-colors|texture-coordinates>` is an optional argument to specify how to output OBJ and PLY meshes mode, for example `--mesh=vertex-colors` or `--mesh=texture-coordinates`. 
`plain` (the default option) outputs the mesh without any color or texture, `vertex-colors` outputs the mesh with colors assigned to every vertex, and `texture-coordinates` will add texture coordinates.

`--no-bundle-adjustment` disables bundle adjustment when reconstructing images with perspective projection.
Adding this flag can significantly reduce processing time, at the cost of producing incorrect data.

`<img1>` and `<img2>` are input filenames for image 1 and 2; supported formats are `jpg`, `tif` and `png`.

`<output>` is the output filename:
* If the filename ends with `.obj`, this will save a 3D [Wavefront OBJ file](https://en.wikipedia.org/wiki/Wavefront_.obj_file).
* If the filename ends with `.ply`, this will save a 3D [PLY binary file](https://en.wikipedia.org/wiki/PLY_(file_format)).
* If the filename ends with `.png`, this will save a PNG depth map file.
* If the filename ends with `.jpg`, this will save a JPEG depth map file.

⚠️ The optimal image size is 1024x1024 (or similar).
Using larger images will result in increased processing times, increased memory usage, and run into GPU hardware limitations.
Smaller images might not have enough details.

### GPU details

Compiling a GPU-accelerated (Vulkan) version requires additional libraries and build tools.

Cybervision was tested to support CPU-only and GPU-accelerated processing on:

* Apple Macbook Air M1 (2020)
* Apple Macbook Pro M1 Max (2021)
* Windows 11, i7-11800H, Geforce RTX 3070 (mobile)
* Fedora CoreOS 37, Celeron N3350 (digital signage appliance)
* Oracle Linux 9, Ampere A1 (Oracle Cloud)

Images up to 4032x3024 should work well, larger images might cause increased memory usage or cause GPU timeouts.

To run Cybervision in Linux, you will need the Vulkan runtime library.
It's called `libvulkan.so.1` and the package is typically called something like `vulkan`, `vulkan-loader`, `libvulkan` or `libvulkan1`.

In Windows and macOS, no additional libraries are required.

## Rust version

Cybervision was rewritten in Rust.

Originally, it was a full all-in-one tool built based on Qt and using a different approach.
For more details about the C++ version, see [Releases](/zlogic/cybervision/releases).
The source code is available in the [branch_qt_sift](../../tree/branch_qt_sift) branch.

The Rust rewrite focuses on the primary goal - generating a 3D surface from an image stereopair;
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

To build it, install Rust and run `cargo build --release`.
