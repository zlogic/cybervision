import logging
import os
import random
import math
import time
from datetime import datetime
from PIL import Image, ImageOps

import cybervision.machine as machine
from cybervision.progressbar import Progressbar
from cybervision.filter import PeakFilter


class NoMatchesFound(Exception):
    def __init__(self, message):
        self.message = message


class Reconstructor:
    def fast_points(self, img: Image):
        return machine.detect(img, self.fast_threshold, self.fast_method, self.fast_nonmax)

    def match_points(self):
        matcher_task = machine.match_start(self.img1, self.img2,
                                           self.points1, self.points2,
                                           self.correlation_kernel_size, self.correlation_threshold,
                                           self.num_threads)
        progressbar = Progressbar()
        while True:
            (completed, percent_complete) = machine.match_status(matcher_task)
            if not completed:
                progressbar.update(percent_complete)
                time.sleep(0.5)
            else:
                return machine.match_result(matcher_task)

    def calculate_model(self, matches):
        sum_dx = 0
        sum_dy = 0
        # TODO: use least squares fitting?
        for match in matches:
            p1 = self.points1[match[0]]
            p2 = self.points2[match[1]]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx**2 + dy**2)
            sum_dx += dx/length
            sum_dy += dy/length
        return sum_dx/len(matches), sum_dy/len(matches)

    def ransac_fit(self):
        suitable_matches = []
        for match in self.matches:
            p1 = self.points1[match[0]]
            p2 = self.points2[match[1]]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            legth = math.sqrt(dx**2 + dy**2)
            if legth > self.ransac_min_length:
                suitable_matches.append(match)

        best_dir_x, best_dir_y = math.nan, math.nan
        best_error = math.inf
        best_matches = []
        for _ in range(self.ransac_k):
            inliers = random.choices(suitable_matches, k=self.ransac_n)
            dir_x, dir_y = self.calculate_model(inliers)
            extended_inliers = []
            for match in suitable_matches:
                if match not in inliers:
                    match_dir_x, match_dir_y = self.calculate_model([match])
                    error = math.sqrt((match_dir_x - dir_x)**2 + (match_dir_y - dir_y)**2)
                    if error < self.ransac_t:
                        extended_inliers.append(match)

            if len(extended_inliers) > self.ransac_d:
                current_matches = inliers + extended_inliers
                current_dir_x, current_dir_y = self.calculate_model(current_matches)
                error = 0
                for match in current_matches:
                    match_dir_x, match_dir_y = self.calculate_model([match])
                    error += math.sqrt((match_dir_x - dir_x)**2 + (match_dir_y - dir_y)**2)/len(current_matches)
                if error < best_error:
                    best_dir_x, best_dir_y = current_dir_x, current_dir_y
                    best_error = error
                    best_matches = current_matches
        return (best_matches, best_dir_x, best_dir_y)

    def create_surface(self):
        correlate_task = machine.correlate_start(self.triangulation_mode, self.img1, self.img2, self.dir_x, self.dir_y,
                                                 self.triangulation_corridor, self.triangulation_kernel_size,
                                                 self.triangulation_threshold,
                                                 self.num_threads, self.triangulation_corridor_segment_length)
        progressbar = Progressbar()
        while True:
            (completed, percent_complete) = machine.correlate_status(correlate_task)
            if not completed:
                progressbar.update(percent_complete)
                time.sleep(0.5)
            else:
                points3d = machine.correlate_result(correlate_task)
                return [(p[0], p[1], p[2]) for p in points3d]

    def filter_peaks(self, width, height):
        filter = PeakFilter(self.points3d, width, height)
        return filter.filter_peaks()

    def reconstruct(self):
        time_started = datetime.now()

        img1_adjusted = ImageOps.autocontrast(self.img1)
        img2_adjusted = ImageOps.autocontrast(self.img2)
        self.points1 = self.fast_points(img1_adjusted)
        self.points2 = self.fast_points(img2_adjusted)
        del(img1_adjusted)
        del(img2_adjusted)

        time_completed_fast = datetime.now()
        self.log.info(f'Extracted points reconstruction in {time_completed_fast-time_started}')
        self.log.info(f'Image 1 has {len(self.points1)} points')
        self.log.info(f'Image 2 has {len(self.points2)} points')

        self.matches = self.match_points()

        if not self.matches:
            raise NoMatchesFound('No matches found')

        time_completed_matching = datetime.now()
        self.log.info(f'Matched keypoints in {time_completed_matching-time_completed_fast}')
        self.log.info(f'Found {len(self.matches)} matches')

        self.matches, self.dir_x, self.dir_y = self.ransac_fit()
        if not self.matches:
            raise NoMatchesFound('Failed to fit the model')

        matches_count = len(self.matches)
        del(self.matches)
        del(self.points1)
        del(self.points2)

        time_completed_ransac = datetime.now()
        self.log.info(f'Completed RANSAC fitting in {time_completed_ransac-time_completed_matching}')
        self.log.info(f'Kept {matches_count} matches')

        if matches_count == 0:
            raise NoMatchesFound('No reliable matches found')

        self.points3d = self.create_surface()
        w1 = self.img1.width
        h1 = self.img1.height
        del(self.img1)
        del(self.img2)

        time_completed_surface = datetime.now()
        self.log.info(f'Completed surface generation in {time_completed_surface-time_completed_ransac}')
        self.log.info(f'Surface contains {len(self.points3d)} points')

        if not self.points3d:
            raise NoMatchesFound('No reliable correlation points found')

        self.points3d = self.filter_peaks(width=w1, height=h1)

        time_completed_filter = datetime.now()
        self.log.info(f'Completed peak filtering in {time_completed_filter-time_completed_surface}')
        self.log.info(f'Surface contains {len(self.points3d)} points')

        time_completed = datetime.now()
        self.log.info(f'Completed reconstruction in {time_completed-time_started}')

    def get_matches(self):
        matches = []
        for m in self.matches:
            p1 = self.points1[m[0]]
            p2 = self.points2[m[1]]
            corr = m[2]
            matches.append((p1[0], p1[1], p2[0], p2[1], corr))
        return matches

    def __init__(self, img1: Image, img2: Image):
        self.img1 = img1.convert('L')
        self.img2 = img2.convert('L')
        self.log = logging.getLogger("reconstructor")
        self.num_threads = os.cpu_count()

        # Tunable parameters
        self.fast_threshold = 15
        self.fast_method = 12
        self.fast_nonmax = True
        self.correlation_threshold = 0.9
        self.correlation_kernel_size = 7
        # slower, but more effective
        # self.correlation_kernel_size = 10
        self.triangulation_kernel_size = 5
        self.triangulation_threshold = 0.8
        self.triangulation_corridor = 5
        self.triangulation_mode = 'gpu'
        # Decrease when using a low-powered GPU
        self.triangulation_corridor_segment_length = 256
        # self.triangulation_corridor = 7
        self.ransac_min_length = 3
        self.ransac_k = 1000
        self.ransac_n = 10
        self.ransac_t = 0.01
        self.ransac_d = 10
