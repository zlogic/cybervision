import logging
import os
import random
import math
from datetime import datetime
from PIL import Image, ImageOps

from cybervision import detect, match

class NoMatchesFound(Exception):
    def __init__(self, message):
        self.message = message

class Reconstructor:
    def fast_points(self, img: Image):
        return detect(img, self.fast_threshold, self.fast_method, self.fast_nonmax)

    def calculate_model(self, matches):
        angle = 0
        # TODO: use least squares fitting?
        for match in matches:
            p1 = self.points1[match[0]]
            p2 = self.points2[match[1]]
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            l = math.sqrt(dx**2 + dy**2)
            if dx>dy:
                angle += math.asin(dy/l) if dx>0 else math.pi-math.asin(dy/l)
            else:
                angle += math.acos(dx/l) if dy>0 else -math.acos(dx/l)
        return angle/len(matches)

    def ransac_fit(self):
        suitable_matches = []
        for match in self.matches:
            p1 = self.points1[match[0]]
            p2 = self.points2[match[1]]
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            l = math.sqrt(dx**2 + dy**2)
            if l>self.ransac_min_length:
                suitable_matches.append(match)
        
        best_model = math.nan
        best_error = math.inf
        best_matches = []
        for _ in range(self.ransac_k):
            inliers = random.choices(suitable_matches, k=self.ransac_n)
            direction = self.calculate_model(inliers)
            extended_inliers = []
            for match in suitable_matches:
                if match not in inliers:
                    match_direction = self.calculate_model([match])
                    error = abs(match_direction - direction)
                    if error < self.ransac_t:
                        extended_inliers.append(match)

            if len(extended_inliers) > self.ransac_d:
                current_matches = inliers + extended_inliers
                current_direction = self.calculate_model(current_matches)
                error = 0
                for match in current_matches:
                    match_direction = self.calculate_model([match])
                    error += abs(match_direction - current_direction)/len(current_matches)
                if error < best_error:
                    best_model = current_direction
                    best_error = error
                    best_matches = current_matches
        return best_matches

    def create_surface(self):
        points3d = []
        for m in self.matches:
            p1 = self.points1[m[0]]
            p2 = self.points2[m[1]]
            #px = (p1[0] + p2[0])/2
            #py = (p1[1] + p2[1])/2
            px = p1[0]
            py = p1[1]
            pz = -math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            points3d.append((px, py, pz))
        return points3d

    def reconstruct(self):
        time_started = datetime.now()

        img1_adjusted = ImageOps.autocontrast(self.img1)
        img2_adjusted = ImageOps.autocontrast(self.img2)
        self.points1 = self.fast_points(img1_adjusted)
        self.points2 = self.fast_points(img2_adjusted)

        time_completed_fast = datetime.now()
        self.log.info(f'Extracted points reconstruction in {time_completed_fast-time_started}')
        self.log.info(f'Image 1 has {len(self.points1)} points')
        self.log.info(f'Image 2 has {len(self.points2)} points')

        self.matches = match(img1_adjusted, img2_adjusted, self.points1, self.points2, self.correlation_kernel_size, self.correlation_threshold, self.num_threads)

        time_completed_matching = datetime.now()
        self.log.info(f'Matched keypoints in {time_completed_matching-time_completed_fast}')
        self.log.info(f'Found {len(self.matches)} matches')

        self.matches = self.ransac_fit()

        time_completed_ransac = datetime.now()
        self.log.info(f'Completed RANSAC fitting in {time_completed_ransac-time_completed_matching}')
        self.log.info(f'Kept {len(self.matches)} matches')

        time_completed = datetime.now()
        self.log.info(f'Completed reconstruction in {time_completed-time_started}')

        self.points3d = self.create_surface()

        if not self.matches:
            raise NoMatchesFound('No reliable matches found')

    def get_matches(self):
        matches = []
        for m in self.matches:
            p1 = self.points1[m[0]]
            p2 = self.points2[m[1]]
            corr = m[2]
            matches.append((p1[0], p1[1], p2[0], p2[1], corr))
        return matches

    def __init__(self, img1: Image, img2: Image):
        self.img1 = img1
        self.img2 = img2
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
        self.ransac_min_length = 5
        self.ransac_k = 1000
        self.ransac_n = 10
        self.ransac_t = 0.01
        self.ransac_d = 10
