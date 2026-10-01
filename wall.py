"""Represent the planar floor or wall used by the cube simulations and draw it
in 3D visualizations. The wall class stores the plane's center, normal vector,
and display size so feature construction and contact calculations can use
the same surface definition. Its geometry helper computes the corners of a
rectangular patch in that plane, and show() adds the patch to a Matplotlib
3D axis. Training, evaluation, and visualization create wall objects to
describe the surface the cube interacts with.
"""

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#This class will store all of the data for a wall or floor for training and evaluating the GNN model
class wall:

    #Saves the size and center position of the wall as well as its normal vector
    def __init__(self,center_position = (0,0,0), size = (1,1), normal = (0,0,1)):
        self.center_position = np.array(center_position)
        self.size = np.array(size)
        self.normal = np.array(normal)

    #Prints out data about the wall
    def __repr__(self):
        return f"Wall(center_position={self.center_position}, size={self.size}, normal={self.normal})"

    #Computes the 4 corner points of the wall for visualization
    def _compute_corners(self):

        tmp = np.array([1, 0, 0]) if abs(self.normal[0]) < 0.9 else np.array([0, 1, 0])

        u = np.cross(self.normal, tmp)
        u = u/np.linalg.norm(u)

        v = np.cross(self.normal, u)

        w, h = self.size / 2

        corners = [
            self.center_position + w*u + h*v,
            self.center_position - w*u + h*v,
            self.center_position - w*u - h*v,
            self.center_position + w*u - h*v
        ]
        return corners

    #Shows the wall on a given 3D axis
    def show(self, ax, color="gray", alpha=0.3):
        """Draw wall on a 3D axis"""
        corners = self._compute_corners()
        poly = Poly3DCollection([corners], color=color, alpha=alpha)
        ax.add_collection3d(poly)
        return poly
