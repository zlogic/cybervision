import argparse
import logging
import sys
from image import SEMImage
from reconstruction import Reconstructor, NoMatchesFound
from visualisation import Visualiser

if __name__ == '__main__':
    logging.basicConfig(level="INFO")

    parser = argparse.ArgumentParser()
    parser.add_argument('img1')
    parser.add_argument('img2')
    args = parser.parse_args()

    img1 = SEMImage(args.img1)
    img2 = SEMImage(args.img2)

    reconstructor = Reconstructor(img1.img, img2.img)

    try:
        reconstructor.reconstruct()
    except NoMatchesFound as err:
        sys.exit(err)
    matches = reconstructor.get_matches()
    
    #v = Visualiser(img1.img, img2.img, reconstructor.get_matches(), reconstructor.points3d)
    #v.show_results()
