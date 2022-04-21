import logging
import os
import random
import math
from datetime import datetime
from PIL import Image, ImageOps
import numpy as np
import scipy.ndimage

from cybervision.machine import detect, match, correlate

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
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            l = math.sqrt(dx**2 + dy**2)
            if abs(dx)>abs(dy):
                angle += math.asin(dy/l) if dx>0 else math.pi-math.asin(dy/l)
            else:
                angle += math.acos(dx/l) if dy>0 else -math.acos(dx/l)
        return angle/len(matches)

    def ransac_fit(self):
        suitable_matches = []
        for match in self.matches:
            p1 = self.points1[match[0]]
            p2 = self.points2[match[1]]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            l = math.sqrt(dx**2 + dy**2)
            if l>self.ransac_min_length:
                suitable_matches.append(match)
        
        best_direction = math.nan
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
                    best_direction = current_direction
                    best_error = error
                    best_matches = current_matches
        return (best_matches, best_direction)

    def create_surface(self):
        # TODO: only angles between pi/4 and 3*pi/4 have been tested, others might require transposing or rotation
        points3d = correlate(self.img1, self.img2, self.angle, self.triangulation_corridor, self.triangulation_kernel_size, self.triangulation_threshold, self.num_threads)
        return [(p[0], p[1], p[2]) for p in points3d]

    def filter_peaks(self):
        depth_grid = np.full((self.img1.width, self.img1.height), np.nan)
        for (x,y,z) in self.points3d:
            depth_grid[x][y] = z

        new_depth_grid = np.full((self.img1.width, self.img1.height), np.nan)
        for y in range(self.img1.height):
            for x in range(self.img1.width):
                z_values = []
                for j in range (-7, 8):
                    y_point = y+j
                    if y_point<0 or y_point>=self.img1.height:
                        continue
                    for i in range (-7, 8):
                        x_point = x+i
                        if x_point<0 or x_point>=self.img1.width:
                            continue
                        z = depth_grid[x_point][y_point]
                        if math.isnan(z):
                            continue
                        z_values.append(z)
                if not z_values or len(z_values) < 5:
                    continue
                median = np.median(z_values)
                stddev = np.std(z_values)
                z = depth_grid[x][y]
                if abs(z-median) < stddev:
                    new_depth_grid[x][y] = z
        depth_grid = new_depth_grid

        return [(x, y, z) for (x,y), z in np.ndenumerate(depth_grid) if not math.isnan(z)]


    def filter_peaks_2(self):
        xy_points = [(p[0], p[1]) for p in self.points3d]
        x_coords = np.linspace(0, self.img1.width, self.img1.width, endpoint=False)
        y_coords = np.linspace(0, self.img1.height, self.img1.height, endpoint=False)
        xx, yy = np.meshgrid(x_coords, y_coords)
        z_values = [p[2] for p in self.points3d]

        depth_grid = np.full((self.img1.width, self.img1.height), np.nan)
        for (x,y,z) in self.points3d:
            depth_grid[x][y] = z
        median = np.nanmedian(depth_grid)
        np.nan_to_num(depth_grid, nan=median)
        depth_grid = scipy.ndimage.percentile_filter(depth_grid, percentile=5, size=64)
        #depth_grid = scipy.ndimage.median_filter(depth_grid, size=10)
        #depth_values = scipy.interpolate.griddata(xy_points, z_values, (xx, yy), method='cubic')
        #depth_values = scipy.ndimage.median_filter(depth_values, size=20)
        #depth_values = scipy.ndimage.percentile_filter(depth_values, percentile=5, size=5)
        #return [(x, y, z) for (x,y,z) in self.points3d if not math.isnan(depth_values[y][x])]
        return [(x, y, z) for (x,y,z) in self.points3d if not math.isnan(depth_grid[x][y])]

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

        self.matches = match(self.img1, self.img2, self.points1, self.points2, self.correlation_kernel_size, self.correlation_threshold, self.num_threads)

        if not self.matches:
            raise NoMatchesFound('No matches found')

        time_completed_matching = datetime.now()
        self.log.info(f'Matched keypoints in {time_completed_matching-time_completed_fast}')
        self.log.info(f'Found {len(self.matches)} matches')

        self.matches, self.angle = self.ransac_fit()
        if not self.matches:
            raise NoMatchesFound('Failed to fit the model')

        time_completed_ransac = datetime.now()
        self.log.info(f'Completed RANSAC fitting in {time_completed_ransac-time_completed_matching}')
        self.log.info(f'Kept {len(self.matches)} matches')

        self.points3d = self.create_surface()

        time_completed_surface = datetime.now()
        self.log.info(f'Completed surface generation in {time_completed_surface-time_completed_ransac}')
        self.log.info(f'Surface contains {len(self.points3d)} points')

        time_completed = datetime.now()
        self.log.info(f'Completed reconstruction in {time_completed-time_started}')
    
        if not self.matches:
            raise NoMatchesFound('No reliable matches found')

        self.points3d = self.filter_peaks()
        #self.points3d = self.filter_peaks_2()

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
        self.triangulation_kernel_size = 5
        self.triangulation_threshold = 0.8
        self.triangulation_corridor = 5
        #self.triangulation_corridor = 7
        self.ransac_min_length = 3
        self.ransac_k = 1000
        self.ransac_n = 10
        self.ransac_t = 0.01
        self.ransac_d = 10
