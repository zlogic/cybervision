import logging
import os
from datetime import datetime
from PIL import Image, ImageOps, ImageDraw

from cybervision import detect, ctx_prepare, match
from image import SEMImage

FAST_THRESHOLD = 15
FAST_METHOD = 12
FAST_NONMAX = True

CORRELATION_KERNEL_SIZE = 5
CORRELATION_THRESHOLD = 0.9

# slower, but more effective
# CORRELATION_KERNEL_SIZE = 10
# CORRELATION_THRESHOLD = 0.8

class Reconstructor:
    def fast_points(self, img: Image):
        return detect(img, FAST_THRESHOLD, FAST_METHOD, FAST_NONMAX)

    def prepare_correlation_context(self, img: Image):
        return ctx_prepare(img, CORRELATION_KERNEL_SIZE, self.num_threads)

    def show_points(self, points1, points2):
        composite = Image.new("RGBA", (self.img1.img.width+self.img2.img.width, max(self.img1.img.height,self.img2.img.height)), (255, 255, 255, 0))
        composite.paste(self.img1.img)
        composite.paste(self.img2.img, (self.img1.img.width, 0))
        draw = ImageDraw.Draw(composite)
        for p in points1:
            draw.point(p, fill=(255, 0, 0, 255))
        for p in points2:
            p = (p[0]+self.img1.img.width, p[1])
            draw.point(p, fill=(255, 0, 0, 255))
        
        composite.show()

    def show_matches(self, points1, points2, matches):
        composite = Image.new("RGBA", (self.img1.img.width+self.img2.img.width, max(self.img1.img.height,self.img2.img.height)), (255, 255, 255, 0))
        composite.paste(self.img1.img)
        composite.paste(self.img2.img, (self.img1.img.width, 0))
        draw = ImageDraw.Draw(composite)

        for m in matches:
            point1 = points1[m[0]]
            point2 = points2[m[1]]
            point2 = (point2[0]+self.img1.img.width, point2[1])
            draw.line(point1 + point2, fill=(255, 255, 255, 255))
            draw.point(point1, fill=(255, 0, 0, 255))
            draw.point(point2, fill=(255, 0, 0, 255))

        composite.show()

    def reconstruct(self):
        time_started = datetime.now()

        img1_adjusted = ImageOps.autocontrast(self.img1.img)
        img2_adjusted = ImageOps.autocontrast(self.img2.img)
        points1 = self.fast_points(img1_adjusted)
        points2 = self.fast_points(img2_adjusted)

        time_completed_fast = datetime.now()
        self.log.info(f'Extracted points reconstruction in {time_completed_fast-time_started}')

        img1_context = self.prepare_correlation_context(self.img1.img)
        img2_context = self.prepare_correlation_context(self.img2.img)

        time_completed_context = datetime.now()
        self.log.info(f'Precomputed correlation data  in {time_completed_context-time_completed_fast}')

        matches = match(img1_context, img2_context, points1, points2, CORRELATION_THRESHOLD, self.num_threads)
        del(img1_context)
        del(img2_context)

        time_completed_matching = datetime.now()
        self.log.info(f'Matched keypoints in {time_completed_matching-time_completed_context}')

        time_completed = datetime.now()
        self.log.info(f'Completed reconstruction in {time_completed-time_started}')

        self.show_points(points1, points2)
        self.show_matches(points1, points2, matches)

    def __init__(self, img1: SEMImage, img2: SEMImage):
        self.img1 = img1
        self.img2 = img2
        self.log = logging.getLogger("reconstructor")
        self.num_threads = os.cpu_count()
