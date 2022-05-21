import sys
import os
from setuptools import setup, Extension
from glob import glob

extra_compile_args = []
sources = [
    'machine/cybervision.c',
    'machine/correlation.c',
    'machine/triangulation.c'
]

if sys.platform in ['darwin', 'linux']:
    extra_compile_args.append('-pthread')
elif sys.platform == 'win32':
    sources.append('machine/win32/pthread.c')

include_dirs = []
library_dirs = []
libraries = []
extra_link_args = []

sdk_path = os.environ.get('VULKAN_SDK')
if sys.platform == 'darwin' and not sdk_path:
    sources.append('machine/metal_correlation.c')
    extra_link_args += ['-framework', 'CoreGraphics', '-framework', 'Metal']

elif not sdk_path:
    raise RuntimeError("VULKAN_SDK is not set")
else:
    sources.append('machine/vulkan_correlation.c')
    library_dirs.append(f'{sdk_path}/lib')
    include_dirs.append(f'{sdk_path}/include')
    if sys.platform == 'darwin':
        include_dirs.append(f'{sdk_path}/libexec/include')
        libraries.append('MoltenVK')
    elif sys.platform == 'linux':
        libraries.append('vulkan')
    elif sys.platform == 'win32':
        libraries.append('vulkan-1')

sources = sources + glob('machine/fast/*.c')
sources = sources + glob('machine/libqhull_r/*.c')

machine = Extension(
    'cybervision.machine',
    sources=sources,
    extra_compile_args=extra_compile_args,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_link_args=extra_link_args
)

setup(
    name='cybervision',
    version='0.3.0',
    python_requires='>=3.8',
    author='Dmitrii Zolotukhin',
    author_email='zlogic@gmail.com',
    description='3D reconstruction software',
    url='https://github.com/zlogic/cybervision',
    packages=['cybervision'],
    license='Apache License, Version 2.0',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    ext_modules=[machine],
    entry_points={
        'console_scripts': [
            'cybervision = cybervision.__main__:main'
        ]
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],
    setup_requires=['wheel'],
    install_requires=[
        "Pillow>=9.1.0"
    ]
)
