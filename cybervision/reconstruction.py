import logging
import os
import random
import math
import time
from datetime import datetime
from PIL import Image, ImageOps

import cybervision.machine as machine
from cybervision.progressbar import Progressbar

class NoMatchesFound(Exception):
    def __init__(self, message):
        self.message = message

class Reconstructor:
    def fast_points(self, img: Image):
        return machine.detect(img, self.fast_threshold, self.fast_method, self.fast_nonmax)

    def match_points(self):
        matcher_task = machine.match_start(self.img1, self.img2, self.points1, self.points2, self.correlation_kernel_size, self.correlation_threshold, self.num_threads)
        progressbar = Progressbar()
        while True:
            (completed, percent_complete) = machine.match_status(matcher_task)
            if not completed:
                progressbar.update(percent_complete)
                time.sleep(0.5)
            else:
                return machine.match_result(matcher_task)

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
        correlate_task = machine.correlate_start(self.img1, self.img2, self.angle, self.triangulation_corridor, self.triangulation_kernel_size, self.triangulation_threshold, self.num_threads)
        progressbar = Progressbar()
        while True:
            (completed, percent_complete) = machine.correlate_status(correlate_task)
            if not completed:
                progressbar.update(percent_complete)
                time.sleep(0.5)
            else:
                points3d = machine.correlate_result(correlate_task)
                return [(p[0], p[1], p[2]) for p in points3d]

    def filter_quad(self, new_quad, current_points):
        keep_points = []
        halfsize = ((new_quad[2]-new_quad[0])/2, (new_quad[3]-new_quad[1])/2)
        center = (new_quad[0]+halfsize[0], new_quad[1]+halfsize[1])
        max_distance = self.filter_quad_radius*(halfsize[0]**2 + halfsize[1]**2)
        for p in current_points:
            distance = (p[0]-center[0])**2 + (p[1]-center[1])**2
            if distance < max_distance:
                keep_points.append(p)
        return (new_quad[0], new_quad[1], new_quad[2], new_quad[3], keep_points)

    def filter_peaks_quad(self, width, height):
        filtered_points = set()
        quadrants = [(0, 0, width, height, self.points3d)]
        iterations = math.floor(math.log(min(width, height)/self.filter_min_size, 2))
        progressbar = Progressbar()
        last_progress_update = datetime.now()
        for i in range(iterations):
            last_iteration = i == iterations-1
            if not quadrants:
                break
            new_quadrants = []
            for (j, q) in enumerate(quadrants):
                quad_points = q[4]
                if not quad_points:
                    continue
                mean_z = 0
                for p in quad_points:
                    mean_z = mean_z+p[2]
                mean_z = mean_z/len(quad_points)
                stdev_z = 0
                for p in quad_points:
                    stdev_z = stdev_z+(p[2]-mean_z)**2
                stdev_z = math.sqrt(stdev_z/len(quad_points))
                qw = int((q[2] - q[0])/2)
                qh = int((q[3] - q[1])/2)
                if stdev_z > self.filter_split_stddev and not last_iteration:
                    new_quadrants.append(self.filter_quad((q[0],    q[1],    q[0]+qw, q[1]+qh), quad_points))
                    new_quadrants.append(self.filter_quad((q[0]+qw, q[1],    q[2],    q[1]+qh), quad_points))
                    new_quadrants.append(self.filter_quad((q[0],    q[1]+qh, q[0]+qw, q[3]   ), quad_points))
                    new_quadrants.append(self.filter_quad((q[0]+qw, q[1]+qh, q[2],    q[3]   ), quad_points))
                else:
                    for p in quad_points:
                        if abs(p[2]-mean_z)<self.filter_match_stddev*stdev_z:
                            filtered_points.add(p)
                
                current_time = datetime.now()
                if (current_time-last_progress_update).total_seconds() > 0.5:
                    last_progress_update = current_time
                    percent_complete = 100.0*(i+j/len(quadrants))/iterations
                    progressbar.update(percent_complete)
            quadrants = new_quadrants

        return list(filtered_points)

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

        self.matches, self.angle = self.ransac_fit()
        if not self.matches:
            raise NoMatchesFound('Failed to fit the model')

        matches_count = len(self.matches)
        if not self.keep_intermediate_results:
            self.matches = []
            self.points1 = []
            self.points2 = []

        time_completed_ransac = datetime.now()
        self.log.info(f'Completed RANSAC fitting in {time_completed_ransac-time_completed_matching}')
        self.log.info(f'Kept {matches_count} matches')
        
        if matches_count == 0:
            raise NoMatchesFound('No reliable matches found')

        self.points3d = self.create_surface()
        w1 = self.img1.width
        h1 = self.img1.height
        if not self.keep_intermediate_results:
            del(self.img1)
            del(self.img2)

        time_completed_surface = datetime.now()
        self.log.info(f'Completed surface generation in {time_completed_surface-time_completed_ransac}')
        self.log.info(f'Surface contains {len(self.points3d)} points')
    
        if not self.points3d:
            raise NoMatchesFound('No reliable correlation points found')

        self.points3d = self.filter_peaks_quad(width=w1, height=h1)

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

    def __init__(self, img1: Image, img2: Image, keep_intermediate_results=False):
        self.img1 = img1
        self.img2 = img2
        self.log = logging.getLogger("reconstructor")
        self.num_threads = os.cpu_count()
        self.keep_intermediate_results = keep_intermediate_results

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
        self.filter_min_size = 16
        self.filter_split_stddev = 1.0
        self.filter_match_stddev = 0.25
        self.filter_quad_radius = 1.5
