import sys
from setuptools import setup, Extension

extra_compile_args = []
sources = [
        'machine/cybervision.c',
        'machine/correlation.c',
        'machine/fast/fast_9.c',
        'machine/fast/fast_10.c',
        'machine/fast/fast_11.c', 
        'machine/fast/fast_12.c',
        'machine/fast/fast.c',
        'machine/fast/nonmax.c'
    ]

if sys.platform in ['darwin', 'linux']:
    extra_compile_args.append('-pthread')
elif sys.platform == 'win32':
    sources.append('machine/win32/pthread.c')

machine = Extension(
    'cybervision.machine',
    sources=sources,
    extra_compile_args=extra_compile_args
)

setup(
    name='cybervision',
    version='0.0.1',
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
        "Pillow>=9.1.0",
        "matplotlib>=3.5.1",
        "scipy>=1.8.0",
        "numpy>=1.22.3"
    ]
)
