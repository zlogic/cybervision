class Visualiser:

    def save_surface_mesh(self, filename):
        xy_points = [(p[0], self.img1.height-p[1]) for p in self.points3d]
        z_values = [p[2] for p in self.points3d]

        mesh = scipy.spatial.Delaunay(xy_points)
        with open(filename, 'w') as f:
            for i, xy in enumerate(xy_points):
                f.write(f'v {xy[0]} {xy[1]} {z_values[i]}\n')

            for t in mesh.simplices:
                point_list = ' '.join([str(tv+1) for tv in t])
                f.write(f'f {point_list}\n')

    def __init__(self, points3d):
        self.points3d = points3d
