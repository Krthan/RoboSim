from RoboticManipulator import Robot
from URDFParser import getJointsAndLinksFromURDF
import numpy as np

urdf_path = "so101_new_calib.urdf"
joints, links = getJointsAndLinksFromURDF(urdf_path)

So101 = Robot("So101", joints, links)
q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
dq = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
print(So101.forward_kinematics(q))
M = So101.getMassMatrixTerm(q)
print(M)
G = So101.getGravityTerm(q)
print(G)
C = So101.getCoriolisTerm(q, dq)
print(C)