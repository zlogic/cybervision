import argparse
import logging
import sys
from cybervision.image import SEMImage
from cybervision.reconstruction import Reconstructor, NoMatchesFound
from cybervision.visualisation import Visualiser


def main():
    logging.basicConfig(level="INFO")

    parser = argparse.ArgumentParser()
    parser.add_argument('img1')
    parser.add_argument('img2')
    parser.add_argument('--resize-scale', type=float, default=1.0)
    args = parser.parse_args()

    img1 = SEMImage(args.img1, args.resize_scale)
    img2 = SEMImage(args.img2, args.resize_scale)

    reconstructor = Reconstructor(img1.img, img2.img)

    try:
        reconstructor.reconstruct()
    except NoMatchesFound as err:
        sys.exit(err)

    v = Visualiser(img1.img, img2.img, reconstructor.get_matches(), reconstructor.points3d)
    v.show_results()


if __name__ == '__main__':
    main()
