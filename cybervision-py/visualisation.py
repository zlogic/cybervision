from PIL import Image, ImageDraw

import numpy as np
from scipy import interpolate, spatial
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

class Visualiser:
    def show_points(self):
        composite = Image.new("RGBA", (self.img1.width+self.img2.width, max(self.img1.height,self.img2.height)), (255, 255, 255, 0))
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
        composite = Image.new("RGBA", (self.img1.width+self.img2.width, max(self.img1.height,self.img2.height)), (255, 255, 255, 0))
        composite.paste(self.img1)
        composite.paste(self.img2, (self.img1.width, 0))
        draw = ImageDraw.Draw(composite)

        for m in self.matches:
            point1 = (m[0], m[1])
            point2 = (m[2]+self.img1.width, m[3])
            draw.line(point1 + point2, fill=(255, 255, 255, 255))
        for m in self.matches:
            draw.point(point1, fill=(255, 0, 0, 255))
            draw.point(point2, fill=(255, 0, 0, 255))

        composite.show()

    def show_distances(self):
        composite = Image.new("RGBA", (self.img1.width+self.img2.width, max(self.img1.height,self.img2.height)), (255, 255, 255, 0))
        composite.paste(self.img1)
        composite.paste(self.img2, (self.img1.width, 0))
        draw = ImageDraw.Draw(composite)

        for m in self.matches:
            point11 = (m[0], m[1])
            point12 = (m[2], m[3])
            point21 = (m[0]+self.img1.width, m[1])
            point22 = (m[2]+self.img1.width, m[3])
            draw.line(point11 + point12, fill=(255, 255, 255, 255))
            draw.line(point21 + point22, fill=(255, 255, 255, 255))
            draw.point(point11, fill=(255, 0, 0, 255))
            draw.point(point12, fill=(255, 0, 0, 255))
            draw.point(point21, fill=(255, 0, 0, 255))
            draw.point(point22, fill=(255, 0, 0, 255))

        composite.show()

    def show_surface_plot(self):
        x_coords = np.linspace(0, self.img1.width, 100)
        y_coords = np.linspace(0, self.img1.height, 100)
        xx, yy = np.meshgrid(x_coords, y_coords)
        xy_coords = [(p[0], p[1]) for p in self.points3d]
        z_values = [p[2] for p in self.points3d]
        interp_grid = interpolate.griddata(xy_coords, z_values, (xx, yy), method='nearest', rescale=True)

        ax = plt.axes(projection='3d')
        ax.plot_surface(xx, yy, interp_grid, shade=True, cmap='jet')
        plt.show()

    def show_surface_mesh(self):
        x = [p[0] for p in self.points3d]
        y = [p[1] for p in self.points3d]
        xy_points = [[p[0], p[1]] for p in self.points3d]
        z_values = [p[2] for p in self.points3d]

        mesh = spatial.Delaunay(xy_points)
        triang = mtri.Triangulation(x=x, y=y, triangles=mesh.vertices)

        ax = plt.axes(projection='3d')
        ax.plot_trisurf(triang, z_values, cmap='jet')
        plt.show()

    def show_surface_image(self):
        surface = Image.new("RGBA", self.img1.size, (255, 0, 255, 255))
        min_z = min(self.points3d, key=lambda p: p[2])[2]
        max_z = max(self.points3d, key=lambda p: p[2])[2]
        draw = ImageDraw.Draw(surface)
        for p in self.points3d:
            z = int((p[2]-min_z)/(max_z-min_z)*255)
            draw.point((p[0], p[1]), fill=(z, z, z, 255))

        surface.show()

    def show_results(self):
        #self.show_points()
        #self.show_matches()
        #self.show_distances()
        self.show_surface_image()
        #self.show_surface_mesh()
        #self.show_surface_plot()

    def __init__(self, img1: Image, img2: Image, matches, points3d):
        self.img1 = img1
        self.img2 = img2
        self.matches = matches
        self.points3d = points3d