import argparse
import logging
from image import SEMImage
from reconstruction import Reconstructor

if __name__ == '__main__':
    logging.basicConfig(level="INFO")

    parser = argparse.ArgumentParser()
    parser.add_argument('img1')
    parser.add_argument('img2')
    args = parser.parse_args()

    img1 = SEMImage(args.img1)
    img2 = SEMImage(args.img2)

    reconstructor = Reconstructor(img1, img2)

    reconstructor.reconstruct()
