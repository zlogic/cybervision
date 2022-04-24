from PIL import Image, ImageDraw

import math
import numpy
import scipy.interpolate
import scipy.spatial

# Sunset from https://jiffyclub.github.io/palettable/cartocolors/sequential/
COLORMAP = [
    [252, 222, 156],
    [250, 164, 118],
    [240, 116, 110],
    [227, 79, 111],
    [220, 57, 119],
    [185, 37, 122],
    [124, 29, 111],
]


class Visualiser:
    def map_color(self, p):
        if p < 0:
            return COLORMAP[0]
        if p >= 1:
            return COLORMAP[-1]
        step = 1/(len(COLORMAP)-2)
        box = math.floor(p/step)
        ratio = (p-step*box)/step
        c1 = COLORMAP[box]
        c2 = COLORMAP[box+1]
        r = int(c2[0]*ratio+c1[0]*(1-ratio))
        g = int(c2[1]*ratio+c1[1]*(1-ratio))
        b = int(c2[2]*ratio+c1[2]*(1-ratio))
        return (r, g, b)

    def save_surface_image(self, filename):
        surface = Image.new("RGBA", self.img1.size, (0, 0, 0, 0))
        min_z = min(self.points3d, key=lambda p: p[2])[2]
        max_z = max(self.points3d, key=lambda p: p[2])[2]
        draw = ImageDraw.Draw(surface)
        for p in self.points3d:
            z = (p[2]-min_z)/(max_z-min_z)
            (r, g, b) = self.map_color(z)
            draw.point((p[0], p[1]), fill=(r, g, b, 255))

        surface.save(filename)

    def save_surface_image_interpolated(self, filename):
        xy_points = [(p[0], p[1]) for p in self.points3d]
        x_coords = numpy.linspace(0, self.img1.width, self.img1.width, endpoint=False)
        y_coords = numpy.linspace(0, self.img1.height, self.img1.height, endpoint=False)
        xx, yy = numpy.meshgrid(x_coords, y_coords)
        z_values = [p[2] for p in self.points3d]
        depth_values = scipy.interpolate.griddata(xy_points, z_values, (xx, yy), method='linear')

        surface = Image.new("RGBA", self.img1.size, (0, 0, 0, 0))
        min_z = numpy.nanmin(depth_values)
        max_z = numpy.nanmax(depth_values)
        for (ix, iy), z in numpy.ndenumerate(depth_values.T):
            if not math.isnan(z):
                x = round(x_coords[ix])
                y = round(y_coords[iy])
                z = (z-min_z)/(max_z-min_z)
                (r, g, b) = self.map_color(z)
                surface.putpixel((x, y), (r, g, b, 255))
        surface.save(filename)

    def __init__(self, img1: Image, img2: Image, points3d):
        self.img1 = img1
        self.img2 = img2
        self.points3d = points3d
