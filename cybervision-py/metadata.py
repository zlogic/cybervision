import argparse
from image import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='append')
    args = parser.parse_args()
    for filename in args.i:
        img = Image(filename)
        print(f'{img.scale_x} {img.scale_y} {img.rotation}')
