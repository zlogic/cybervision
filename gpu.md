# Vulkan

## MoltenVK

Download [molten-vk](https://formulae.brew.sh/formula/molten-vk) from Homebrew.

To set a custom logging level, use the `MVK_CONFIG_LOG_LEVEL` evironment variable (e.g. `export MVK_CONFIG_LOG_LEVEL=NONE`).

## Compile shaders

Download the [glslang](https://github.com/KhronosGroup/glslang) compiler from [Homebrew](https://formulae.brew.sh/formula/glslang).

Compile the shader and convert it into a C source that can be embedded into the binary.

```shell
glslangValidator -V shaders/correlation.glsl -o shaders/correlation.spv
xxd -i shaders/correlation.spv > machine/shaders_spv.h
rm shaders/correlation.spv
```

Or with the MoltenVKShaderConverter:

```shell
MoltenVKShaderConverter -gi shaders/correlation.glsl -oh -t c -so machine/shaders_spv.h
```
