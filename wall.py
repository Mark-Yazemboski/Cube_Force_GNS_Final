import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#This class will store all of the data for a wall or floor for training and evaluating the GNN model
class wall:

    #Saves the size and center position of the wall as well as its normal vector
    def __init__(self,center_position = (0,0,0), size = (1,1), normal = (0,0,1)):
        self.center_position = np.array(center_position)
        self.size = np.array(size)
        self.normal = np.array(normal)
    
    #Given a point it will calculate the distance from that point to the wall along the wall normal
    def get_distance_to_point(self, point):
        point = np.array(point)
        point_to_center = point - self.center_position
        distance = np.dot(point_to_center, self.normal)
        return distance
    
    #Prints out data about the wall
    def __repr__(self):
        return f"Wall(center_position={self.center_position}, size={self.size}, normal={self.normal})"
    
    #Returns the wall normal vector
    def get_normal(self):
        return self.normal

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
    
    #Updates the position and size of the wall object
    def update_position_width(self, new_center_position=None, new_size=None):
        """Update wall position and/or size"""
        if new_center_position is not None:
            self.center_position = np.array(new_center_position)
        if new_size is not None:
            self.size = np.array(new_size)

    #Shows the wall on a given 3D axis
    def show(self, ax, color="gray", alpha=0.3):
        """Draw wall on a 3D axis"""
        corners = self._compute_corners()
        poly = Poly3DCollection([corners], color=color, alpha=alpha)
        ax.add_collection3d(poly)
        return poly
