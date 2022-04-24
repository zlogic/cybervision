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
    parser.add_argument('--interpolate', dest='interpolate', action='store_true')
    parser.add_argument('--no-interpolate', dest='interpolate', action='store_false')
    parser.set_defaults(interpolate=True)
    parser.add_argument('--output-file', required=True)
    args = parser.parse_args()

    img1 = SEMImage(args.img1, args.resize_scale)
    img2 = SEMImage(args.img2, args.resize_scale)

    reconstructor = Reconstructor(img1.img, img2.img)

    try:
        reconstructor.reconstruct()
    except NoMatchesFound as err:
        sys.exit(err)

    v = Visualiser(img1.img, img2.img, reconstructor.points3d)
    if args.interpolate:
        v.save_surface_image_interpolated(args.output_file)
    else:
        v.save_surface_image(args.output_file)


if __name__ == '__main__':
    main()
