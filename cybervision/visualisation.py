from PIL import Image, ImageDraw

import math
import numpy as np
import scipy.interpolate
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

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

    def show_points(self):
        size = (self.img1.width+self.img2.width, max(self.img1.height, self.img2.height))
        composite = Image.new("RGBA", size, (255, 255, 255, 0))
        composite.paste(self.img1)
        composite.paste(self.img2, (self.img1.width, 0))
        draw = ImageDraw.Draw(composite)
        for m in self.matches:
            point1 = (m[0], m[1])
            point2 = (m[2]+self.img1.width, m[3])
            draw.point(point1, fill=(255, 0, 0, 255))
            draw.point(point2, fill=(255, 0, 0, 255))
        composite.show()

    def show_matches(self):
        size = (self.img1.width+self.img2.width, max(self.img1.height, self.img2.height))
        composite = Image.new("RGBA", size, (255, 255, 255, 0))
        composite.paste(self.img1)
        composite.paste(self.img2, (self.img1.width, 0))
        draw = ImageDraw.Draw(composite)

        for m in self.matches:
            point1 = (m[0], m[1])
            point2 = (m[2]+self.img1.width, m[3])
            draw.line(point1 + point2, fill=(255, 0, 0, 255))
        for m in self.matches:
            draw.point(point1, fill=(0, 255, 0, 255))
            draw.point(point2, fill=(0, 255, 0, 255))

        composite.show()

    def show_distances(self):
        size = (self.img1.width+self.img2.width, max(self.img1.height, self.img2.height))
        composite = Image.new("RGBA", size, (255, 255, 255, 0))
        composite.paste(self.img1)
        composite.paste(self.img2, (self.img1.width, 0))
        draw = ImageDraw.Draw(composite)

        for m in self.matches:
            point11 = (m[0], m[1])
            point12 = (m[2], m[3])
            point21 = (m[0]+self.img1.width, m[1])
            point22 = (m[2]+self.img1.width, m[3])
            draw.line(point11 + point12, fill=(255, 0, 0, 255))
            draw.line(point21 + point22, fill=(255, 0, 0, 255))
            draw.point(point11, fill=(0, 255, 0, 255))
            draw.point(point12, fill=(0, 255, 0, 255))
            draw.point(point21, fill=(0, 255, 0, 255))
            draw.point(point22, fill=(0, 255, 0, 255))

        composite.show()

    def show_surface_plot(self):
        x_coords = np.linspace(0, self.img1.width, self.img1.width, endpoint=False)
        y_coords = np.linspace(0, self.img1.height, self.img1.height, endpoint=False)
        xx, yy = np.meshgrid(x_coords, y_coords)
        xy_coords = [(p[0], p[1]) for p in self.points3d]
        z_values = [p[2] for p in self.points3d]
        interp_grid = scipy.interpolate.griddata(xy_coords, z_values, (xx, yy), method='linear')

        ax = plt.axes(projection='3d')
        ax.plot_surface(xx, yy, interp_grid, rcount=100, ccount=100, shade=True, cmap='jet')
        plt.show()

    def show_surface_mesh(self):
        # Too slow without resampling
        x = [p[0] for p in self.points3d]
        y = [p[1] for p in self.points3d]
        xy_points = [[p[0], p[1]] for p in self.points3d]
        z_values = [p[2] for p in self.points3d]

        mesh = scipy.spatial.Delaunay(xy_points)
        triang = mtri.Triangulation(x=x, y=y, triangles=mesh.vertices)

        ax = plt.axes(projection='3d')
        ax.plot_trisurf(triang, z_values, cmap='jet', shade=False)
        plt.show()

    def show_surface_image(self):
        surface = Image.new("RGBA", self.img1.size, (0, 0, 0, 0))
        min_z = min(self.points3d, key=lambda p: p[2])[2]
        max_z = max(self.points3d, key=lambda p: p[2])[2]
        draw = ImageDraw.Draw(surface)
        for p in self.points3d:
            z = (p[2]-min_z)/(max_z-min_z)
            (r, g, b) = self.map_color(z)
            draw.point((p[0], p[1]), fill=(r, g, b, 255))

        surface.show()

    def show_surface_image_interpolated(self):
        xy_points = [(p[0], p[1]) for p in self.points3d]
        x_coords = np.linspace(0, self.img1.width, self.img1.width, endpoint=False)
        y_coords = np.linspace(0, self.img1.height, self.img1.height, endpoint=False)
        xx, yy = np.meshgrid(x_coords, y_coords)
        z_values = [p[2] for p in self.points3d]
        depth_values = scipy.interpolate.griddata(xy_points, z_values, (xx, yy), method='linear')

        surface = Image.new("RGBA", self.img1.size, (0, 0, 0, 0))
        min_z = np.nanmin(depth_values)
        max_z = np.nanmax(depth_values)
        for (ix, iy), z in np.ndenumerate(depth_values.T):
            if not math.isnan(z):
                x = round(x_coords[ix])
                y = round(y_coords[iy])
                z = (z-min_z)/(max_z-min_z)
                (r, g, b) = self.map_color(z)
                surface.putpixel((x, y), (r, g, b, 255))
        surface.show()

    def show_results(self):
        # self.show_points()
        # self.show_matches()
        # self.show_distances()
        # self.show_surface_plot()
        # self.show_surface_mesh()
        self.show_surface_image()
        self.show_surface_image_interpolated()

    def __init__(self, img1: Image, img2: Image, matches, points3d):
        self.img1 = img1
        self.img2 = img2
        self.matches = matches
        self.points3d = points3d
