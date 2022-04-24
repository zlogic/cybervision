import math
from datetime import datetime

from cybervision.progressbar import Progressbar


class PeakFilter:

    def filter_quad(self, new_quad, current_points):
        keep_points = []
        halfsize = ((new_quad[2]-new_quad[0])/2, (new_quad[3]-new_quad[1])/2)
        center = (new_quad[0]+halfsize[0], new_quad[1]+halfsize[1])
        max_distance = self.quad_radius*(halfsize[0]**2 + halfsize[1]**2)
        for p in current_points:
            distance = (p[0]-center[0])**2 + (p[1]-center[1])**2
            if distance < max_distance:
                keep_points.append(p)
        return (new_quad[0], new_quad[1], new_quad[2], new_quad[3], keep_points)

    def process_quadrant(self, q, last_iteration: bool):
        new_quadrants = []
        quad_points = q[4]
        if not quad_points:
            return
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
        if stdev_z > self.split_stddev and not last_iteration:
            new_quadrants.append(self.filter_quad((q[0], q[1], q[0]+qw, q[1]+qh), quad_points))
            new_quadrants.append(self.filter_quad((q[0]+qw, q[1], q[2], q[1]+qh), quad_points))
            new_quadrants.append(self.filter_quad((q[0], q[1]+qh, q[0]+qw, q[3]), quad_points))
            new_quadrants.append(self.filter_quad((q[0]+qw, q[1]+qh, q[2], q[3]), quad_points))
            return new_quadrants
        else:
            for p in quad_points:
                if abs(p[2]-mean_z) < self.match_stddev*stdev_z:
                    self.filtered_points.add(p)
        return

    def filter_peaks(self):
        self.filtered_points = set()
        quadrants = self.quadrants
        iterations = self.iterations
        progressbar = Progressbar()
        last_progress_update = datetime.now()
        for i in range(iterations):
            last_iteration = i == iterations-1
            if not quadrants:
                break
            new_quadrants = []
            for (j, q) in enumerate(quadrants):
                quad_new_quadrants = self.process_quadrant(q, last_iteration)
                if quad_new_quadrants:
                    new_quadrants = new_quadrants + quad_new_quadrants

                current_time = datetime.now()
                if (current_time-last_progress_update).total_seconds() > 0.5:
                    last_progress_update = current_time
                    percent_complete = 100.0*(i+j/len(quadrants))/iterations
                    progressbar.update(percent_complete)
            quadrants = new_quadrants

        return list(self.filtered_points)

    def __init__(self, points3d, width, height):
        # Tunable parameters
        self.min_size = 16
        self.split_stddev = 1.0
        self.match_stddev = 0.25
        self.quad_radius = 1.5

        self.iterations = math.floor(math.log(min(width, height)/self.min_size, 2))
        self.quadrants = [(0, 0, width, height, points3d)]
