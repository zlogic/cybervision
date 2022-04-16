import logging
import os
from datetime import datetime
from PIL import Image, ImageOps, ImageDraw

import machine
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
        return machine.detect(img, FAST_THRESHOLD, FAST_METHOD, FAST_NONMAX)

    def prepare_correlation_context(self, img: Image):
        return machine.ctx_prepare(img, CORRELATION_KERNEL_SIZE, self.num_threads)

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

        matches = machine.match(img1_context, img2_context, points1, points2, CORRELATION_THRESHOLD, self.num_threads)
        del(img1_context)
        del(img2_context)

        time_completed_matching = datetime.now()
        self.log.info(f'Matched keypoints in {time_completed_matching-time_completed_context}')

        time_completed = datetime.now()
        self.log.info(f'Completed reconstruction in {time_completed-time_started}')

        composite = Image.new("RGBA", (img1_adjusted.width+img2_adjusted.width, max(img1_adjusted.height,img2_adjusted.height)), (255, 255, 255, 0))
        composite.paste(self.img1.img)
        composite.paste(self.img2.img, (self.img1.img.width, 0))
        draw = ImageDraw.Draw(composite)

        for m in matches:
            point1 = points1[m[0]]
            point2 = points2[m[1]]
            draw.line(point1 + (point2[0] + self.img1.img.width, point2[1]), fill=(255, 255, 255, 255))

        composite.show()


    def __init__(self, img1: SEMImage, img2: SEMImage):
        self.img1 = img1
        self.img2 = img2
        self.log = logging.getLogger("reconstructor")
        self.num_threads = os.cpu_count()
