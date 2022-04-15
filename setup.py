from setuptools import setup, Extension

machine = Extension('machine',sources=[
    'machine/fast/fast_9.c',
    'machine/fast/fast_10.c',
    'machine/fast/fast_11.c',
    'machine/fast/fast_12.c',
    'machine/fast/fast.c',
    'machine/fast/nonmax.c',
    'machine/machine.c'
    ])

setup(
    name='cybervision',
    version='0.0.1',
    python_requires='>=3.8',
    author='Dmitrii Zolotukhin',
    author_email='zlogic@gmail.com',
    description='3D reconstruction software',
    url='https://github.com/zlogic/cybervision',
    packages=['cybervision-py'],
    license='Apache License, Version 2.0',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=[machine],
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
