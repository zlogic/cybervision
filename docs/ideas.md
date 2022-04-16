# Improvement ideas

Some ideas for future improvement, which may or may not be implemented in the fugure

## Matching FAST corners

Use dual-side matching:

* first, search for matches from image A to image B
* then, search for matches from image B to image A

## GPU acceleration

Depending on the target platform, try to use Vulkan or Metal/MoltenVK on Apple systems.

* https://github.com/baldand/py-metal-compute
* https://github.com/pygfx/wgpu-py
* Or call Metal/Vulkan directly from C code
* Or use a wrapper library like glfw

## 3D viewer

* Export .obj files and use an [online viewer](https://3dviewer.net) or any other compatible viewer.

Look into existing viewers:

* https://github.com/pygfx/wgpu-py
* https://github.com/gabdube/python-vulkan-triangle or https://github.com/gabdube/panic-panda
* https://github.com/ikalevatykh/panda3d_viewer
* imgui or wxwidgets with a 3D viewer widget

## Binary builds

Use Github actions to build wheels for Windows and macOS.

Windows:

* add pthread support through `vcpkg.exe install pthread` (link with pthread-win32)
* create pthread.h (pthread.c) stub to implement missing methods
