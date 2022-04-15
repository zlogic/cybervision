from setuptools import setup, Extension

fast = Extension('fast',sources=[
    'fast/fast_9.c',
    'fast/fast_10.c',
    'fast/fast_11.c',
    'fast/fast_12.c',
    'fast/fast.c',
    'fast/nonmax.c',
    'fast/fast_python.c'
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
    ext_modules=[fast],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],
    setup_requires=['wheel']
)
