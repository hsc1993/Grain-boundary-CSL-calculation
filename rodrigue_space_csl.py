# importing libraries
import numpy as np
import math
from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import ast  # to inteperate string into list
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


######### ######### #########
# rule of rotation for either point or matrix
# rotation = rodrigues_rotation(axis_test, angle)
# point = rotation.dot(point)
# matrix = (matrix.dot(rotation.transpose()))

######### ######### #########


def degreefy(radian):
    return radian * 180 / math.pi


def radianfy(degree):
    return degree / 180 * math.pi


def crystal_director_to_lab(R_tensor, crystal_x, crystal_y, crystal_z):
    lab_x = R_tensor.dot(crystal_x)
    lab_y = R_tensor.dot(crystal_y)
    lab_z = R_tensor.dot(crystal_z)
    return lab_x, lab_y, lab_z


def crystal_sequential_to_lab(R_x, R_y, R_z, crystal_x, crystal_y, crystal_z):
    lab_x = R_x.dot(R_y.dot(R_z.dot(crystal_x)))
    lab_y = R_x.dot(R_y.dot(R_z.dot(crystal_y)))
    lab_z = R_x.dot(R_y.dot(R_z.dot(crystal_z)))
    return lab_x, lab_y, lab_z


def lab_director_to_crystal(R_tensor, lab_x, lab_y, lab_z):
    crystal_x = R_tensor.transpose().dot(lab_x)
    crystal_y = R_tensor.transpose().dot(lab_y)
    crystal_z = R_tensor.transpose().dot(lab_z)
    return crystal_x, crystal_y, crystal_z


def lab_sequential_to_crystal(R_x, R_y, R_z, lab_x, lab_y, lab_z):
    crystal_x = R_z.transpose().dot(R_y.transpose().dot(R_x.transpose().dot(lab_x)))
    crystal_y = R_z.transpose().dot(R_y.transpose().dot(R_x.transpose().dot(lab_y)))
    crystal_z = R_z.transpose().dot(R_y.transpose().dot(R_x.transpose().dot(lab_z)))
    return crystal_x, crystal_y, crystal_z


def rodrigues_rotation(axis, rotation):
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R_return = np.identity(3) + K.dot(np.sin(rotation)) + (K.dot(K)).dot((1 - np.cos(rotation)))
    return R_return


def extract_angle(R):
    cos_rotation = (R.trace() - 1) / 2
    if cos_rotation > 1:
        cos_rotation = 1
    if cos_rotation < -1:
        cos_rotation = -1
    return np.arccos(cos_rotation)


def extract_axis(R):
    axis = [(R[1, 2] - R[2, 1]) / 2,
            (R[2, 0] - R[0, 2]) / 2,
            (R[0, 1] - R[1, 0]) / 2]

    axis = axis / LA.norm(axis)
    return axis


def extract_axis_rotation(R):
    if (R.trace() - 1) / 2 > 1:
        angle = np.arccos(1)
    elif (R.trace() - 1) / 2 < -1:
        angle = np.arccos(-1)
    else:
        angle = extract_angle(R)

    # special case for no rotation or 360 degree
    if abs(angle) < 0.001:
        temp = np.zeros((3, 1))
        for i in range(len(temp)):
            temp[i] = R[i, i] + 1
            if temp[i] < 0:
                temp[i] = 0

        axis = np.array([math.sqrt(temp[0] / 2),
                         math.sqrt(temp[1] / 2),
                         math.sqrt(temp[2] / 2), ])
        axis = axis / LA.norm(axis)
        angle = 0
        # print('no rotation')
        return axis, angle

    # special case for 180 degree
    if abs(angle - math.pi) < 0.001:
        temp = np.zeros((3, 1))
        for i in range(len(temp)):
            temp[i] = R[i, i] + 1
            if temp[i] < 0:
                temp[i] = 0

        axis = np.array([math.sqrt(temp[0] / 2),
                         math.sqrt(temp[1] / 2),
                         math.sqrt(temp[2] / 2), ])
        axis = axis / LA.norm(axis)
        angle = math.pi
        return axis, angle

    # usual case
    axis = [(R[1, 2] - R[2, 1]) / math.sin(angle) / 2,
            (R[2, 0] - R[0, 2]) / math.sin(angle) / 2,
            (R[0, 1] - R[1, 0]) / math.sin(angle) / 2]

    axis = axis / LA.norm(axis)

    return axis, angle


def symmetry_operator(i):
    simple_crystal_x = np.array([[1, 0, 0]])
    simple_crystal_y = np.array([[0, 1, 0]])
    simple_crystal_z = np.array([[0, 0, 1]])

    # simple_crystal_x = np.array([ref[0]])
    # simple_crystal_y = np.array([ref[1]])
    # simple_crystal_z = np.array([ref[2]])
    simple_crystal_x = simple_crystal_x / LA.norm(simple_crystal_x)
    simple_crystal_y = simple_crystal_y / LA.norm(simple_crystal_y)
    simple_crystal_z = simple_crystal_z / LA.norm(simple_crystal_z)
    simple_crystal = np.concatenate((simple_crystal_x, simple_crystal_y, simple_crystal_z), axis=0)

    if (i < 9):  # 4 fold symmetry for simple cubic with 3 axes
        rotation = (90 * (i % 3 + 1)) / 180 * math.pi
        if (i < 3):
            axis = simple_crystal[0]
        if (3 <= i & i < 6):
            axis = simple_crystal[1]
        if (6 <= i & i < 9):
            axis = simple_crystal[2]

    if (9 <= i & i < 17):  # 3 fold symmetry for simple cubic with 4 axes
        rotation = (120 * (i % 2 + 1)) / 180 * math.pi
        if (i < 11):
            axis = + simple_crystal[0] + simple_crystal[1] + simple_crystal[2]
        if (11 <= i & i < 13):
            axis = - simple_crystal[0] + simple_crystal[1] + simple_crystal[2]
        if (13 <= i & i < 15):
            axis = + simple_crystal[0] - simple_crystal[1] + simple_crystal[2]
        if (15 <= i & i < 17):
            axis = + simple_crystal[0] + simple_crystal[1] - simple_crystal[2]

    if (17 <= i & i < 23):  # two fold symmetry for simple cubic with 6 axes
        rotation = 180 / 180 * math.pi
        if (i == 17):
            axis = simple_crystal[0] - simple_crystal[1]
        if (i == 18):
            axis = simple_crystal[0] + simple_crystal[1]
        if (i == 19):
            axis = simple_crystal[1] - simple_crystal[2]
        if (i == 20):
            axis = simple_crystal[1] + simple_crystal[2]
        if (i == 21):
            axis = simple_crystal[0] - simple_crystal[2]
        if (i == 22):
            axis = simple_crystal[0] + simple_crystal[2]
    if (i == 23):  # identity
        rotation = 360 / 180 * math.pi
        axis = simple_crystal[1]

    axis = axis / LA.norm(axis)
    R_return = rodrigues_rotation(axis, rotation)
    return R_return


#
# def stereographic_projection(pole_nd, point):
#     south_to_north = 2 * (pole_nd)
#     negative_to_axis = point - (-pole_nd)
#     south_to_north_norm = LA.norm(south_to_north)
#     negative_to_axis_norm = LA.norm(negative_to_axis)
#
#     arccos_theta_half = negative_to_axis.dot(south_to_north) / south_to_north_norm / negative_to_axis_norm
#     if arccos_theta_half > 1:
#         arccos_theta_half = 1
#     if arccos_theta_half < -1:
#         arccos_theta_half = -1
#
#     theta = 2 * np.arccos(arccos_theta_half)
#
#     frac = (pole_nd[0] * point[0] + pole_nd[1] * point[1] + pole_nd[2] * point[2]) / (
#             pole_nd[0] * pole_nd[0] + pole_nd[1] * pole_nd[1] + pole_nd[2] * pole_nd[2])
#     axis_normalline_pole = frac * pole_nd
#     axis_projected = point - axis_normalline_pole
#
#     if abs(axis_projected[0]) < 0.001:
#         phi = 0
#     else:
#         phi = np.arctan(axis_projected[1] / axis_projected[0])
#
#     if point[0] == 0:
#         px = 0
#         py = 0
#
#     else:
#         if point[2] > 1:
#             point[2] = 1
#         elif point[2] < -1:
#             point[2] = -1
#
#         # theta, phi = stereographic_projection(pole_nd, point)
#         px = np.array([np.tan(theta / 2) * np.cos(phi)])
#         py = np.array([np.tan(theta / 2) * np.sin(phi)])
#
#     return px, py


def disorientation(crystal1_ref, crystal2_ref, labplot_ref):
    rotation_crystal1_to_labplot = labplot_ref.dot(crystal1_ref.transpose())
    for i in range(24):
        symmetry_operator_1 = symmetry_operator(i)
        for j in range(24):
            symmetry_operator_2 = symmetry_operator(j)

            misorientation_1to2 = crystal2_ref.dot(crystal1_ref.transpose())
            misorientation_2to1 = crystal1_ref.dot(crystal2_ref.transpose())

            disorientation_1to2 = symmetry_operator_1.dot(misorientation_1to2.dot(symmetry_operator_2.transpose()))
            disorientation_2to1 = symmetry_operator_1.dot(misorientation_2to1.dot(symmetry_operator_2.transpose()))

            # matrix = (matrix.dot(rotation.transpose()))

            for switch in range(2):
                if switch == 0:
                    disorientation_chosen = disorientation_1to2
                if switch == 1:
                    disorientation_chosen = disorientation_2to1

                axis, angle = extract_axis_rotation(disorientation_chosen)

                if angle == 0:
                    axis = crystal1_ref[2]

                # print('i,j=', i, j)
                # print('axis=', axis)
                # print('angle=', angle)
                if i == 0 & j == 0:
                    angle_min = angle
                    axis_min = axis
                elif angle < angle_min:
                    angle_min = angle
                    axis_min = axis

    return axis_min, angle_min


def stereographic_projection(pole_rd, pole_nd, point):
    point = point / LA.norm(point)
    pole_nd = pole_nd / LA.norm(pole_nd)
    south_to_north = 2 * (pole_nd)
    south_to_point = point - (-pole_nd)
    south_to_north_norm = LA.norm(south_to_north)
    south_to_point_norm = LA.norm(south_to_point)

    arccos_theta_half = south_to_point.dot(south_to_north) / south_to_north_norm / south_to_point_norm
    if arccos_theta_half > 1:
        arccos_theta_half = 1
    if arccos_theta_half < -1:
        arccos_theta_half = -1

    theta = 2 * np.arccos(arccos_theta_half)
    frac = (pole_nd[0] * point[0] + pole_nd[1] * point[1] + pole_nd[2] * point[2]) / (
            pole_nd[0] * pole_nd[0] + pole_nd[1] * pole_nd[1] + pole_nd[2] * pole_nd[2])
    point_normalline_pole = frac * pole_nd
    point_projected = point - point_normalline_pole

    if LA.norm(point_projected) == 0:
        # projection of northpole is singular
        arccos_phi = 0
    else:
        arccos_phi = point_projected.dot(pole_rd) / LA.norm(point_projected) / LA.norm(pole_rd)
    if arccos_phi > 1:
        arccos_phi = 1
    if arccos_phi < -1:
        arccos_phi = -1

    phi = np.arccos(arccos_phi)

    if point[0] == 0:
        px = 0
        py = 0
    else:
        px = np.array([np.tan(theta / 2) * np.cos(phi)])
        py = np.array([np.tan(theta / 2) * np.sin(phi)])

    return px, py


def rotate_pole_by_euler(phi2, psi, axis_td, axis_nd):
    axis1 = axis_td
    axis2 = axis_nd

    axis_probe = axis_nd

    R1 = rodrigues_rotation(axis1, psi)
    R2 = rodrigues_rotation(axis2, phi2)

    point_temp = R1.dot(axis_probe)
    point = R2.dot(point_temp)
    return point


def three_fold_boundary_phi2_to_psi(phi2):
    cos_phi2 = np.cos(phi2)
    cos_psi = cos_phi2 / np.sqrt(1 + pow(cos_phi2, 2))
    psi = np.arccos(cos_psi)
    return psi


def endpoints_unittriangle(pole_td, pole_nd):
    # a,b,twist in standard lab frame
    phi2_b = math.pi / 4
    psi_b = three_fold_boundary_phi2_to_psi(phi2_b)
    phi2_a = 0
    psi_a = three_fold_boundary_phi2_to_psi(phi2_a)
    phi2_twist = 0
    psi_twist = 0

    # lab_b, lab_a, lab_twist are standard [001],[101],[111]
    point_twist_lab = rotate_pole_by_euler(phi2_twist, psi_twist, pole_td, pole_nd)
    point_a_lab = rotate_pole_by_euler(phi2_a, psi_a, pole_td, pole_nd)
    point_b_lab = rotate_pole_by_euler(phi2_b, psi_b, pole_td, pole_nd)
    return point_twist_lab, point_a_lab, point_b_lab


def radian_between_two_points(point_1, point_2):
    point_1 = point_1 / LA.norm(point_1)
    point_2 = point_2 / LA.norm(point_2)
    arccos_angle_a_to_b = point_1.dot(point_2) / LA.norm(point_1) / LA.norm(point_2)
    if arccos_angle_a_to_b > 1:
        arccos_angle_a_to_b = 1
    if arccos_angle_a_to_b < -1:
        arccos_angle_a_to_b = -1
    return np.arccos(arccos_angle_a_to_b)


def cubic_FZ_threefold_boundary_pointlist():
    # angle_a_to_b is the phi2 angle measured from a to b, not the exact rotation angle
    angle_a_to_b = math.pi / 4
    phi2_list = [val for val in np.arange(0, angle_a_to_b, 0.01)]
    psi_list = []
    for phi2 in phi2_list:
        psi = three_fold_boundary_phi2_to_psi(phi2)
        psi_list.append(psi)
    return phi2_list, psi_list


def fun_cubic_FZ(px_test, py_test):
    outside = 0

    df = pd.read_csv('px_py.csv')
    df_px_py = pd.read_csv('px_py.csv')
    df_px = df_px_py['px']
    df_py = df_px_py['py']
    px = df_px.to_numpy()
    py = df_py.to_numpy()
    px_a, py_a = px[0], py[0]
    px_b, py_b = px[-1], py[-1]
    px_twist, py_twist = 0, 0

    # boundary from twist to a
    if (px_test > px_a) | (px_test < px_twist) | (py_test > py_b) | (py_test < py_twist):
        outside = 1
        # print('boundary from twist to a')

    # boundary from a to b
    else:
        df_sort = df.iloc[(df['py'] - py_test).abs().argsort()[1]]
        px_from_py_test = df_sort['px']
        if px_from_py_test < px_test:
            outside = 1
            # print('boundary from a to b')

    # boundary from twist to b
    if (px_test > px_twist) & (px_test < px_b) & (py_test / px_test > py_b / px_b):
        outside = 1
        # print('boundary from twist to b')

    # plot_FZ(px_test, py_test, outside,  pole_rd, pole_td, pole_nd)
    return outside


def create_pxpy_csv(pole_rd_input, pole_td_input, pole_nd_input):
    phi2_list, psi_list = cubic_FZ_threefold_boundary_pointlist()

    point_probe = pole_nd_input
    point_probe = point_probe / LA.norm(point_probe)
    num_points = len(phi2_list)
    point_pole_list = []
    for i in range(num_points):
        phi2 = phi2_list[i]
        psi = psi_list[i]
        R1 = rodrigues_rotation(pole_td_input, psi)
        R2 = rodrigues_rotation(pole_nd_input, phi2)
        point_temp = R1.dot(point_probe)
        point_pole = R2.dot(point_temp)
        point_pole_list.append(point_pole)
    # generating px-py pair to form the boundary
    for i in range(num_points):
        px_ele, py_ele = stereographic_projection(pole_rd_input, pole_nd_input, point_pole_list[i])
        px_py_ele = [[px_ele.item(), py_ele.item()]]
        if i == 0:
            px = px_ele
            py = py_ele
            px_py_list = px_py_ele
        else:
            px = np.append(px, px_ele, axis=0)
            py = np.append(py, py_ele, axis=0)
            px_py_list = np.append(px_py_list, px_py_ele, axis=0)
    # px, py = stereographic_projection(pole_rd, pole_nd, axis_list[0])

    # writing data to csv
    df_px = pd.DataFrame({'px': px})
    df_py = pd.DataFrame({'py': py})
    df_px_py = pd.concat([df_px, df_py], axis=1)
    df_px_py.to_csv('px_py.csv')


def plot_FZ(pole_rd_input, pole_td_input, pole_nd_input, ax):
    create_pxpy_csv(pole_rd_input, pole_td_input, pole_nd_input)
    df_px_py = pd.read_csv('px_py.csv')
    df_px = df_px_py['px']
    df_py = df_px_py['py']
    px = df_px.to_numpy()
    py = df_py.to_numpy()
    px_a, py_a = px[0], py[0]
    px_b, py_b = px[-1], py[-1]

    px_twist, py_twist = 0, 0

    line_a_x = [px_twist, px_a]
    line_a_y = [py_twist, py_a]
    line_b_x = [px_twist, px_b]
    line_b_y = [py_twist, py_b]
    point_twist_x = px_twist
    point_twist_y = py_twist
    ax.plot(line_a_x, line_a_y, linewidth=3)
    ax.plot(line_b_x, line_b_y, linewidth=3)
    ax.scatter(px_a, py_a, linewidth=3)
    ax.scatter(px_b, py_b, linewidth=3)
    ax.scatter(point_twist_x, point_twist_y, linewidth=3)
    ax.plot(px, py)


class Lattice_site:
    def __init__(self, triplets):
        self.coordinates = triplets
        self.index = None
        self.x_idx = None
        self.y_idx = None
        self.index_xy = None
        self.distance_to_axis = None
        self.neighbors = []
        self.from_grain = None

        # make distinction between ex. fcc's face center and the rest
        self.on_principle_axis = False

        # once defined within a plane, it has x,y value
        self.x = None
        self.y = None


class Plane:
    def __init__(self, axis, crystal_object, layer_index):
        self.axis = axis / LA.norm(axis)
        self.origin = np.array([0, 0, 0])

        self.x_lab = np.array([1, 0, 0])
        self.y_lab = np.array([0, 1, 0])
        self.z_lab = np.array([0, 0, 1])

        # e1 e2 e3 are used to generate grid among planar lattice sites
        self.e3 = self.axis

        # axis in x,y plane
        self.lattice = []
        for i in range(len(crystal_object)):
            # all lattice sites on the plane should be perpendicular to the given axis
            if np.isclose((crystal_object[i].coordinates - axis.dot(layer_index)).dot(axis), 0):
                self.lattice.append(crystal_object[i])

        # planar distance from lattice site to axis, used to calculate sigma
        for idx, site in enumerate(self.lattice):
            self.lattice[idx].index = idx
            self.lattice[idx].distance_to_axis = LA.norm(self.lattice[idx].coordinates)

        # extract principle lattice coordinates ie. ignoring face centers for fcc
        self.extract_principle_lattice_coordinates()

        # using principle lattice coordinates to extract rolling, transverse directions
        self.extract_rolling_transverse_from_principle_lattice_coordinates()
        # generate planar coordinates in crystal(planar) frame, ie. rolling,transver plane
        for site in self.lattice:
            site.x, site.y = self.generate_planar_coordinates(site.coordinates, self.rolling, self.transverse)

        self.generate_index()

        self.generate_neighbors()

        # rolling,transverse = extract_rolling_transverse_from_planar_coordinates()
        self.ref = np.array([self.rolling, self.transverse, self.axis])

    def extract_principle_lattice_coordinates(self):
        self.principle_lattice_coordinates = []
        self.principle_lattice_sites = []
        for idx, site in enumerate(self.lattice):
            if self.lattice[idx].on_principle_axis:
                self.principle_lattice_coordinates.append(self.lattice[idx].coordinates.tolist())
                self.principle_lattice_sites.append(self.lattice[idx])

    def extract_rolling_transverse_from_principle_lattice_coordinates(self):
        # first find origin for the plane
        for site in self.lattice:
            if np.array_equal(site.coordinates, self.origin):
                planar_origin = site

        # then find the neighbors of origin for later rolling direction calculation
        # for idx, potential_neighbor in enumerate(self.principle_lattice_sites):
        for idx, potential_neighbor in enumerate(self.lattice):
            planar_distance = LA.norm(potential_neighbor.coordinates - planar_origin.coordinates)
            if planar_distance == 0:
                continue
            if idx == 0:
                shortest_planar_distance = planar_distance
            elif planar_distance < shortest_planar_distance:
                shortest_planar_distance = planar_distance

        # calculate rolling and transverse directions
        # for potential_neighbor in self.principle_lattice_sites:
        rolling_direction_neighbor_list = []
        for idx, potential_neighbor in enumerate(self.lattice):
            planar_distance = LA.norm(potential_neighbor.coordinates - planar_origin.coordinates)
            if np.isclose(np.round(planar_distance, 3), np.around(shortest_planar_distance, 3)):
                rolling_direction_neighbor_list.append(potential_neighbor)

        rolling_direction_neighbor = rolling_direction_neighbor_list[0]
        self.rolling = rolling_direction_neighbor.coordinates / LA.norm(rolling_direction_neighbor.coordinates)
        self.transverse = np.cross(self.axis, self.rolling)
        self.e1 = self.rolling
        # only four neighbors means the e1 e2 are perpendicular
        if len(rolling_direction_neighbor_list) == 4:
            self.e2 = self.transverse
        else:
            for e2_direction_neighbor in rolling_direction_neighbor_list[1:]:
                cos_rolling_e2 = e2_direction_neighbor.coordinates.dot(self.rolling) / LA.norm(
                    e2_direction_neighbor.coordinates) / LA.norm(self.rolling)
                cos_transverse_e2 = e2_direction_neighbor.coordinates.dot(self.transverse) / LA.norm(
                    e2_direction_neighbor.coordinates) / LA.norm(self.transverse)

                if (cos_rolling_e2 > 0) & (cos_transverse_e2 > 0):
                    self.e2 = e2_direction_neighbor.coordinates / LA.norm(e2_direction_neighbor.coordinates)

        #
        # rolling = np.array([self.axis[1], -self.axis[0], 0])
        # if np.allclose(rolling, 0):
        #     rolling = np.array([1, 0, 0])
        # transverse = np.array([self.axis[0] * self.axis[2], self.axis[1] * self.axis[2],
        #                        - self.axis[0] * self.axis[0] - self.axis[1] * self.axis[1]])
        #
        # rolling = rolling / LA.norm(rolling)
        # transverse = transverse / LA.norm(transverse)

    def generate_planar_coordinates(self, lattice_coordinate, rolling, transverse):
        distance_to_axis = LA.norm(np.array(lattice_coordinate))
        if np.isclose(distance_to_axis, 0):
            x = 0
            y = 0
        else:
            # using angle between axis and lattice site to calculate planar coordinate
            cos_angle_wrt_rolling = np.array(lattice_coordinate).dot(rolling) / LA.norm(
                np.array(lattice_coordinate)) / LA.norm(rolling)
            if cos_angle_wrt_rolling > 1:
                cos_angle_wrt_rolling = 1
            if cos_angle_wrt_rolling < -1:
                cos_angle_wrt_rolling = -1
            angle_wrt_rolling = np.arccos(cos_angle_wrt_rolling)

            cos_angle_wrt_transverse = np.array(lattice_coordinate).dot(transverse) / LA.norm(
                np.array(lattice_coordinate)) / LA.norm(transverse)
            if cos_angle_wrt_transverse > 1:
                cos_angle_wrt_transverse = 1
            if cos_angle_wrt_transverse < -1:
                cos_angle_wrt_transverse = -1
            angle_wrt_transverse = np.arccos(cos_angle_wrt_transverse)

            x = distance_to_axis * np.cos(angle_wrt_rolling)
            y = distance_to_axis * np.cos(angle_wrt_transverse)
        return x, y

    def generate_neighbors(self):
        # first find origin for the plane
        for site in self.lattice:
            if np.array_equal(site.coordinates, self.origin):
                planar_origin = site

        # then calculate distance from origin to its neighbors
        for idx, site in enumerate(self.lattice):
            distance = LA.norm(site.coordinates - planar_origin.coordinates)
            if distance == 0:
                continue
            if idx == 0:
                distance_origin_to_neightbor = distance
            elif distance < distance_origin_to_neightbor:
                distance_origin_to_neightbor = distance

        for site in self.lattice:
            for neighbor_candidate in self.lattice:
                if abs(LA.norm(site.index_xy - neighbor_candidate.index_xy)) == 1:
                    site.neighbors.append(neighbor_candidate)
            # for neighbor in site.neighbors:
            #     print('neighbor=',neighbor.index_xy)
            # print()

            # if LA.norm(site.coordinates - planar_origin.coordinates) == distance_origin_to_neightbor:
            #     site.neighbors.append(neighbor_candidate)
            # for neighbor_candidate in site.neighbors:
            #     print('distance_origin_to_neightbor=',distance_origin_to_neightbor)
            #     print('site.neighbors=',neighbor_candidate.coordinates)
            # print()

    def express_planar_coordinates_in_e1_e2(self, x, y, e1_planar, e2_planar):
        X = np.array([[e1_planar[0], e1_planar[1]],
                      [e2_planar[0], e2_planar[1]]])
        A = np.array([x, y])
        Xinv = LA.inv(X)
        b = A.dot(Xinv)
        x_e1e2 = b[0]
        y_e1e2 = b[1]
        return x_e1e2, y_e1e2

    def generate_index(self):
        # if plane is shifted, origin needs to be adjusted
        origin_on_plane_x, origin_on_plane_y = self.generate_planar_coordinates(self.origin, self.rolling,
                                                                                self.transverse)
        e1x, e1y = self.generate_planar_coordinates(self.e1, self.rolling, self.transverse)
        e2x, e2y = self.generate_planar_coordinates(self.e2, self.rolling, self.transverse)

        e1_planar = np.array([e1x, e1y])
        e2_planar = np.array([e2x, e2y])

        for site in self.lattice:
            site.x_e1e2, site.y_e1e2 = self.express_planar_coordinates_in_e1_e2(site.x, site.y, e1_planar, e2_planar)

        # calculate minimum distance along e1 and e2 axis
        points_on_e1 = []
        points_on_e2 = []
        for idx, site in enumerate(self.lattice):
            points_on_e1.append(np.round(site.x_e1e2, 3))
            points_on_e2.append(np.round(site.y_e1e2, 3))

        points_on_e1.sort()
        points_on_e2.sort()
        points_on_e1 = list(dict.fromkeys(points_on_e1))
        points_on_e2 = list(dict.fromkeys(points_on_e2))

        step_e1 = points_on_e1[1] - points_on_e1[0]
        step_e2 = points_on_e2[1] - points_on_e2[0]

        self.e1_planar = e1_planar.dot(LA.norm(step_e1))
        self.e2_planar = e2_planar.dot(LA.norm(step_e2))

        for idx, site in enumerate(self.lattice):
            x_planar = (self.lattice[idx].x - origin_on_plane_x) / step_e1
            y_planar = (self.lattice[idx].y - origin_on_plane_y) / step_e2

            self.lattice[idx].x_idx, self.lattice[idx].y_idx = self.express_planar_coordinates_in_e1_e2(x_planar,
                                                                                                        y_planar,
                                                                                                        e1_planar,
                                                                                                        e2_planar)
            self.lattice[idx].x_idx = np.round(self.lattice[idx].x_idx, 0)
            self.lattice[idx].y_idx = np.round(self.lattice[idx].y_idx, 0)
            self.lattice[idx].index_xy = np.array(
                [self.lattice[idx].x_idx, self.lattice[idx].y_idx])

    def shift(self, shift):
        shift_projected_x, shift_projected_y = self.generate_planar_coordinates(shift, self.rolling,
                                                                                self.transverse)
        self.origin = self.origin + shift
        for idx, lattice_site in enumerate(self.lattice):
            self.lattice[idx].coordinates = self.lattice[idx].coordinates + shift
            self.lattice[idx].x, self.lattice[idx].y = self.generate_planar_coordinates(self.lattice[idx].coordinates,
                                                                                        self.rolling, self.transverse)

        shift_projected_in_e1e2 = [shift_projected_x, shift_projected_y]
        return shift_projected_in_e1e2

    def rotate(self, axis, angle):
        axis = axis / LA.norm(axis)
        rotation = rodrigues_rotation(axis, angle)

        for idx, lattice_site in enumerate(self.lattice):
            coordinate_temp = rotation.dot(self.lattice[idx].coordinates)
            self.lattice[idx].coordinates = coordinate_temp
            self.lattice[idx].x, self.lattice[idx].y = self.generate_planar_coordinates(self.lattice[idx].coordinates,
                                                                                        self.rolling, self.transverse)

    def lattice_coordinates(self):
        coordinates = []
        for site in self.lattice:
            coordinate = site.coordinates
            coordinates.append(coordinate)
        return coordinates

    def hkl(self):
        # calculating hkl
        h = self.axis[0]
        k = self.axis[1]
        l = self.axis[2]

        # find the least integer combination for given axis
        while (abs(h) - abs(int(h)) > 0.01) | (abs(k) - abs(int(k)) > 0.01) | (abs(l) - abs(int(l)) > 0.01):
            h, k, l = h * 10, k * 10, l * 10
        h, k, l = int(h), int(k), int(l)
        gcd_hk = math.gcd(h, k)
        gcd_hkl = math.gcd(gcd_hk, l)
        h, k, l = h / gcd_hkl, k / gcd_hkl, l / gcd_hkl
        self.h = h
        self.k = k
        self.l = l
        self.hkl = np.array([h, k, l])


class Crystal:
    def __init__(self, lattice_type, upper_limit):
        lattice_range = list(range(-upper_limit, upper_limit + 1))
        u1 = list([1, 0, 0])
        u2 = list([0, 1, 0])
        u3 = list([0, 0, 1])
        u1 = np.array(u1)
        u2 = np.array(u2)
        u3 = np.array(u3)
        self.lattice = []

        triplets = list(itertools.product(lattice_range, repeat=3))
        triplets = np.array(triplets)
        if lattice_type == 'sc':
            for i in range(len(triplets)):
                lattice_site = Lattice_site(triplets[i])
                self.lattice.append(lattice_site)

        if lattice_type == 'fcc':
            for i in range(len(triplets)):
                lattice_site = Lattice_site(triplets[i])
                self.lattice.append(lattice_site)

            # shift all coordinates by u2/2+u3/2, effectively creating face centers
            triplets1 = np.array(triplets)
            triplets1 = np.add(triplets1, u2 / 2 + u3 / 2)
            for i in range(len(triplets1)):
                lattice_site = Lattice_site(triplets1[i])
                if site_in_the_box(lattice_site, upper_limit):
                    self.lattice.append(lattice_site)

            triplets2 = np.array(triplets)
            triplets2 = np.add(triplets2, u1 / 2 + u3 / 2)
            for i in range(len(triplets2)):
                lattice_site = Lattice_site(triplets2[i])
                if site_in_the_box(lattice_site, upper_limit):
                    self.lattice.append(lattice_site)

            triplets3 = np.array(triplets)
            triplets3 = np.add(triplets3, u1 / 2 + u2 / 2)
            for i in range(len(triplets3)):
                lattice_site = Lattice_site(triplets3[i])
                if site_in_the_box(lattice_site, upper_limit):
                    self.lattice.append(lattice_site)

            for lattice_site in self.lattice:
                if np.isclose(lattice_site.coordinates[0], round(lattice_site.coordinates[0], 0)) & \
                        np.isclose(lattice_site.coordinates[1], round(lattice_site.coordinates[1], 0)) & \
                        np.isclose(lattice_site.coordinates[2], round(lattice_site.coordinates[2], 0)):
                    lattice_site.on_principle_axis = True

        if lattice_type == 'bcc':
            triplets1 = list(itertools.product(lattice_range, repeat=3))
            triplets1 = np.array(triplets1)
            triplets1 = np.add(triplets1, u1 / 2 + u2 / 2 + u3 / 2)

            for i in range(len(triplets1)):
                lattice_site = Lattice_site(triplets1[i])
                self.lattice.append(lattice_site)

            for i in range(len(triplets)):
                lattice_site = Lattice_site(triplets[i])
                self.lattice.append(lattice_site)

    def lattice_coordinates(self):
        coordinates = []
        for site in self.lattice:
            coordinate = site.coordinates
            coordinates.append(coordinate)
        return coordinates

    def shift(self, direction, distance):
        shift = direction * distance
        for idx, lattice_site in enumerate(self.lattice):
            self.lattice[idx].coordinates = self.lattice[idx].coordinates + shift

    def rotate(self, axis, angle):
        rotation = rodrigues_rotation(axis, angle)
        for idx, lattice_site in enumerate(self.lattice):
            self.lattice[idx].coordinates = rotation.dot(self.lattice[idx].coordinates)


class DSC_site:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.from_grain = None
        self.neighbors = []



class DSC:
    # It was the lattice of all translations of one of the crystals that conserved the given CSL
    def __init__(self, grain1, grain2):
        self.lattice = []
        self.burgers_vectors = []

        # generate complete dsc lattice based on given two grains
        xcoordinates_dsc = []
        ycoordinates_dsc = []
        for site1 in grain1.lattice:
            if site1.x not in xcoordinates_dsc:
                xcoordinates_dsc.append(site1.x)
            if site1.y not in ycoordinates_dsc:
                ycoordinates_dsc.append(site1.y)
        for site2 in grain2.lattice:
            if site2.x not in xcoordinates_dsc:
                xcoordinates_dsc.append(site2.x)
            if site2.y not in ycoordinates_dsc:
                ycoordinates_dsc.append(site2.y)

        for x in xcoordinates_dsc:
            for y in ycoordinates_dsc:
                site = DSC_site(x, y)
                self.lattice.append(site)

        for site in self.lattice:
            for site1 in grain1.lattice:
                if (site1.x == site.x) & (site1.y == site.y):
                    site.from_grain = 1
            for site2 in grain1.lattice:
                if (site2.x == site.x) & (site2.y == site.y):
                    site.from_grain = 2

        # copy grain1's attributes to DSC lattice
        self.rolling = grain1.rolling
        self.transverse = grain1.transverse
        self.axis = grain1.axis
        self.origin = grain1.origin
        self.e1 = grain1.e1
        self.e2 = grain1.e2


        self.generate_index()



        # self.generate_neighbors()
        # for site in self.lattice:
        #     for neighbor in site.neighbors:
        #         print('neighbor.x=',neighbor.x)

        for site1 in grain1.lattice:
            for site2 in grain2.lattice:
                displacement_from_grain1_to_grain2 = np.round(site2.coordinates - site1.coordinates, 3)
                if np.allclose(displacement_from_grain1_to_grain2, 0):
                    continue
                self.burgers_vectors.append(site2.coordinates - site1.coordinates)

        self.burgers_vectors = sorted(self.burgers_vectors, key=lambda x: LA.norm(x))
        self.norm_burgers = [np.round(LA.norm(burger), 3) for burger in self.burgers_vectors]
        self.norm_burgers = [i for n, i in enumerate(self.norm_burgers) if i not in self.norm_burgers[:n]]
        if 0.0 in self.norm_burgers:
            self.norm_burgers.remove(0.0)

        grain1_lattice_coordinates = []
        grain2_lattice_coordinates = []
        for site1 in grain1.lattice:
            grain1_lattice_coordinates.append(site1.coordinates)
        for site2 in grain2.lattice:
            grain2_lattice_coordinates.append(site2.coordinates)

        self.csl_sites = []
        csl_sites_coordinates = []
        for grain1_lattice_coordinate in grain1_lattice_coordinates:
            for grain2_lattice_coordinate in grain2_lattice_coordinates:
                if np.array_equal(np.round(grain1_lattice_coordinate, 3), np.round(grain2_lattice_coordinate, 3)):
                    csl_sites_coordinates.append(np.round(grain1_lattice_coordinate, 3))

        for site in grain1.lattice:
            for csl_sites_coordinate in csl_sites_coordinates:
                if np.array_equal(np.round(site.coordinates, 3), csl_sites_coordinate):
                    self.csl_sites.append(site)

    def generate_planar_coordinates(self, lattice_coordinate, rolling, transverse):
        distance_to_axis = LA.norm(np.array(lattice_coordinate))
        if np.isclose(distance_to_axis, 0):
            x = 0
            y = 0
        else:
            # using angle between axis and lattice site to calculate planar coordinate
            cos_angle_wrt_rolling = np.array(lattice_coordinate).dot(rolling) / LA.norm(
                np.array(lattice_coordinate)) / LA.norm(rolling)
            if cos_angle_wrt_rolling > 1:
                cos_angle_wrt_rolling = 1
            if cos_angle_wrt_rolling < -1:
                cos_angle_wrt_rolling = -1
            angle_wrt_rolling = np.arccos(cos_angle_wrt_rolling)

            cos_angle_wrt_transverse = np.array(lattice_coordinate).dot(transverse) / LA.norm(
                np.array(lattice_coordinate)) / LA.norm(transverse)
            if cos_angle_wrt_transverse > 1:
                cos_angle_wrt_transverse = 1
            if cos_angle_wrt_transverse < -1:
                cos_angle_wrt_transverse = -1
            angle_wrt_transverse = np.arccos(cos_angle_wrt_transverse)

            x = distance_to_axis * np.cos(angle_wrt_rolling)
            y = distance_to_axis * np.cos(angle_wrt_transverse)
        return x, y

    # def generate_neighbors(self):
        # first find origin for the plane
        # planar_origin = self.origin
        #
        # # then calculate distance from origin to its neighbors
        # for idx, site in enumerate(self.lattice):
        #     distance = LA.norm(site.coordinates - planar_origin.coordinates)
        #     if distance == 0:
        #         continue
        #     if idx == 0:
        #         distance_origin_to_neightbor = distance
        #     elif distance < distance_origin_to_neightbor:
        #         distance_origin_to_neightbor = distance

        # for site in self.lattice:
            # for neighbor_candidate in self.lattice:
            # site.neighbors.append(neighbor_candidate)




    def express_planar_coordinates_in_e1_e2(self, x, y, e1_planar, e2_planar):
        X = np.array([[e1_planar[0], e1_planar[1]],
                      [e2_planar[0], e2_planar[1]]])
        A = np.array([x, y])
        Xinv = LA.inv(X)
        b = A.dot(Xinv)
        x_e1e2 = b[0]
        y_e1e2 = b[1]
        return x_e1e2, y_e1e2

    def generate_index(self):
        # if plane is shifted, origin needs to be adjusted
        origin_on_plane_x, origin_on_plane_y = self.generate_planar_coordinates(self.origin, self.rolling,
                                                                                self.transverse)
        e1x, e1y = self.generate_planar_coordinates(self.e1, self.rolling, self.transverse)
        e2x, e2y = self.generate_planar_coordinates(self.e2, self.rolling, self.transverse)

        e1_planar = np.array([e1x, e1y])
        e2_planar = np.array([e2x, e2y])

        for site in self.lattice:
            site.x_e1e2, site.y_e1e2 = self.express_planar_coordinates_in_e1_e2(site.x, site.y, e1_planar,
                                                                                e2_planar)

        # calculate minimum distance along e1 and e2 axis
        points_on_e1 = []
        points_on_e2 = []
        for idx, site in enumerate(self.lattice):
            points_on_e1.append(np.round(site.x_e1e2, 3))
            points_on_e2.append(np.round(site.y_e1e2, 3))

        points_on_e1.sort()
        points_on_e2.sort()
        points_on_e1 = list(dict.fromkeys(points_on_e1))
        points_on_e2 = list(dict.fromkeys(points_on_e2))

        step_e1 = points_on_e1[1] - points_on_e1[0]
        step_e2 = points_on_e2[1] - points_on_e2[0]

        self.e1_planar = e1_planar.dot(LA.norm(step_e1))
        self.e2_planar = e2_planar.dot(LA.norm(step_e2))
        for idx, site in enumerate(self.lattice):
            x_planar = (self.lattice[idx].x - origin_on_plane_x) / step_e1
            y_planar = (self.lattice[idx].y - origin_on_plane_y) / step_e2

            self.lattice[idx].x_idx, self.lattice[idx].y_idx = self.express_planar_coordinates_in_e1_e2(x_planar,
                                                                                                        y_planar,
                                                                                                        e1_planar,
                                                                                                        e2_planar)
            self.lattice[idx].x_idx = np.round(self.lattice[idx].x_idx, 0)
            self.lattice[idx].y_idx = np.round(self.lattice[idx].y_idx, 0)
            self.lattice[idx].index_xy = np.array(
                [self.lattice[idx].x_idx, self.lattice[idx].y_idx])




def site_in_the_box(lattice_site, upper_limit):
    if np.abs(lattice_site.coordinates[0]) > upper_limit:
        return False
    if np.abs(lattice_site.coordinates[1]) > upper_limit:
        return False
    if np.abs(lattice_site.coordinates[2]) > upper_limit:
        return False
    return True


def generate_crystal_matrix(lattice_type, upper_limit):
    lattice_range = list(range(-upper_limit, upper_limit + 1))
    u1 = list([1, 0, 0])
    u2 = list([0, 1, 0])
    u3 = list([0, 0, 1])
    u1 = np.array(u1)
    u2 = np.array(u2)
    u3 = np.array(u3)

    triplets = list(itertools.product(lattice_range, repeat=3))
    triplets = np.array(triplets)
    if lattice_type == 'sc':
        lattice = []
        for i in range(len(triplets)):
            lattice.append(Lattice_site(triplets[i]))
        return lattice

    if lattice_type == 'fcc':
        lattice = []
        triplets1 = list(itertools.product(lattice_range, repeat=3))
        triplets1 = np.array(triplets1)
        triplets1 = np.add(triplets1, u2 / 2 + u3 / 2)
        for i in range(len(triplets1)):
            lattice_site = Lattice_site(triplets1[i])
            if site_in_the_box(lattice_site, upper_limit):
                lattice.append(lattice_site)

        triplets2 = list(itertools.product(lattice_range, repeat=3))
        triplets2 = np.array(triplets2)
        triplets2 = np.add(triplets2, u1 / 2 + u3 / 2)
        for i in range(len(triplets2)):
            lattice_site = Lattice_site(triplets2[i])
            if site_in_the_box(lattice_site, upper_limit):
                lattice.append(lattice_site)

        triplets3 = list(itertools.product(lattice_range, repeat=3))
        triplets3 = np.array(triplets3)
        triplets3 = np.add(triplets3, u1 / 2 + u2 / 2)
        for i in range(len(triplets3)):
            lattice_site = Lattice_site(triplets3[i])
            if site_in_the_box(lattice_site, upper_limit):
                lattice.append(lattice_site)

        for i in range(len(triplets)):
            lattice.append(Lattice_site(triplets[i]))
        return lattice

    if lattice_type == 'bcc':
        triplets1 = list(itertools.product(lattice_range, repeat=3))
        triplets1 = np.array(triplets1)
        triplets1 = np.add(triplets1, u1 / 2 + u2 / 2 + u3 / 2)

        lattice = []

        for i in range(len(triplets1)):
            lattice_site = Lattice_site(triplets1[i])
            if site_in_the_box(lattice_site, upper_limit):
                lattice.append(lattice_site)

        for i in range(len(triplets)):
            lattice.append(Lattice_site(triplets[i]))

        return lattice


def check_axis_family(axis1, axis2):
    axis1 = np.absolute(axis1)
    axis2 = np.absolute(axis2)
    axis1 = axis1.tolist()
    axis2 = axis2.tolist()

    count = 0
    for element1 in axis1:
        for element2 in axis2:
            if np.isclose(element1, element2):
                count = count + 1

    flag_family = 0
    if count >= 3:
        flag_family = 1
    return flag_family


def sigma_generate(crystal_matrix, axis_list):
    # creating lattice
    labplot_ref = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # plot axis and two adjacent planes
    sigma_data = []
    sigma_round_data = []
    angle_data = []
    uvw_data = []
    axis_index_data = []
    angle_disorientation_data = []
    uvw_list_data = []
    first_site_coordinates_data = []
    second_site_coordinates_data = []
    # chose an axis to generate layer
    for axis_index, axis in enumerate(axis_list):
        axis = axis / LA.norm(axis)
        print('axis=', axis)
        layer = Plane(axis, crystal_matrix, 0)

        list_radius = []
        for site in layer.lattice:
            normal_check = axis.dot(site.coordinates)
            if normal_check != 0:
                print('not normal')

            list_radius.append(site.distance_to_axis.tolist())
        list_radius_unique = np.unique(list_radius, axis=0).tolist()

        # remove origin which is trivial
        for radius in list_radius_unique:
            num_csl = list_radius.count(radius)
            if num_csl < 3:
                list_radius_unique.remove(radius)

        # for each distance, choose i site and j site to calculate sigma and angle
        # first, choose a unique radius
        for radius_unique in list_radius_unique:
            csl_sites = []
            csl_sites_coordinates = []
            # second, choose a site which has that unique distance to axis
            for site in layer.lattice:
                if np.isclose(site.distance_to_axis, radius_unique):
                    csl_sites.append(layer.lattice[site.index])
                    csl_sites_coordinates.append(layer.lattice[site.index].coordinates)
            # third, calculate all possible distances(angles) between two sites of the same radius
            for first_site in csl_sites:
                # choosing the second site
                distance_among_sites = []
                for site in csl_sites:
                    distance_among_sites.append(LA.norm(site.coordinates - first_site.coordinates))
                distance_among_sites.sort()
                distance_among_sites.remove(0)
                if len(csl_sites) < 2:
                    continue

                for site in csl_sites:
                    if LA.norm(site.coordinates - first_site.coordinates) == distance_among_sites[0]:
                        second_site = site
                        first_site_coordinates = first_site.coordinates.tolist()
                        second_site_coordinates = second_site.coordinates.tolist()

                vector_m2 = (second_site.coordinates - first_site.coordinates) / 2
                vector_m1 = (second_site.coordinates + first_site.coordinates) / 2
                if (not (np.isclose(np.array(vector_m2).dot(axis), 0))) | (
                        not (np.isclose(np.array(vector_m1).dot(axis), 0))):
                    print('m1,m2 not perpendicular to axis')
                m2 = LA.norm(vector_m2)
                m1 = LA.norm(vector_m1)
                sigma = m1 ** 2 + m2 ** 2
                sigma = sigma * 4

                sigma_round = round(sigma)

                if sigma_round == 0:
                    continue
                while sigma_round % 2 == 0:
                    sigma_round = sigma_round / 2
                if sigma_round == 1:
                    continue

                if m1 == 0:
                    angle = math.pi
                else:
                    angle = 2 * math.atan(m2 / m1)

                if angle > 62.8 * math.pi / 180:
                    layer_rotation = rodrigues_rotation(axis, angle)
                    layer_rotated_rolling = layer_rotation.dot(layer.rolling)
                    layer_rotated_transverse = layer_rotation.dot(layer.transverse)
                    layer_rotated_axis = layer_rotation.dot(layer.axis)
                    layer_rotated_ref = np.array(
                        [layer_rotated_rolling, layer_rotated_transverse, layer_rotated_axis])
                    axis_disorientation, angle_disorientation = disorientation(layer.ref, layer_rotated_ref,
                                                                               labplot_ref)

                    flag_family = check_axis_family(axis_disorientation, axis)
                    # if flag_family == 1:
                    # print('same family')
                    # if flag_family == 0:
                    # print('not the same family')
                    # continue
                else:
                    angle_disorientation = angle

                sigma_data.append(sigma)
                sigma_round_data.append(sigma_round)
                uvw_list = axis_list[axis_index].tolist()
                uvw_list_data.append(uvw_list)
                uvw_string = ''.join([str(elem) for elem in uvw_list])
                uvw_int = int(uvw_string)
                uvw_data.append(uvw_int)
                axis_index_data.append(axis_index)
                angle_data.append(round(angle * 180 / math.pi, 2))
                angle_disorientation_data.append(round(angle_disorientation * 180 / math.pi, 2))

                first_site_coordinates_data.append(first_site_coordinates)
                second_site_coordinates_data.append(second_site_coordinates)

    uvw_float = []
    for uvw in uvw_list_data:
        uvw_float_temp = [float(elem) for elem in uvw]
        uvw_float_temp = uvw_float_temp / LA.norm(uvw_float_temp)
        uvw_float.append(uvw_float_temp)

    angle_list = [angle * math.pi / 180 for angle in angle_data]

    rVec = []
    for idx, uvw in enumerate(uvw_float):
        rVec_temp = [math.tan(angle_list[idx] / 2) * item for item in uvw]
        rVec_temp = [np.round(element, 3) for element in rVec_temp]
        rVec.append(rVec_temp)

    df_csl = pd.DataFrame(
        np.array([sigma_data, sigma_round_data, angle_data, angle_disorientation_data, uvw_data, uvw_list_data,
                  rVec, first_site_coordinates_data, second_site_coordinates_data]).transpose(),
        columns=['sigma', 'sigma_round', 'angle', 'angle_disorientation', 'uvw', 'uvw_list', 'rVec', 'first_site',
                 'second_site'])

    # find lowest disorientation
    df_csl = find_lowest_disorientation(df_csl)
    # df_csl.to_csv('csl.csv')
    return df_csl


def find_lowest_disorientation(df_csl):
    df_csl = df_csl[df_csl.angle_disorientation != 0]
    df_csl = df_csl.drop_duplicates(subset=['sigma_round', 'angle_disorientation', 'uvw'], keep='first')
    df_csl = df_csl.sort_values(by=['sigma_round', 'angle_disorientation'])
    df_csl = df_csl[df_csl['sigma_round'] < 36]
    df_csl = df_csl.reset_index(drop=True)
    return df_csl


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def main():
    # setting up rodrigues space three axes, also vertices in SST
    pole_rd = np.array([0, 1, 0])
    pole_td = np.array([0, 0, 1])
    pole_nd = np.array([1, 0, 0])
    pole_rd = pole_rd / LA.norm(pole_rd)
    pole_td = pole_td / LA.norm(pole_td)
    pole_nd = pole_nd / LA.norm(pole_nd)

    # using the the three axes to generate SST boundary and write into csv file
    create_pxpy_csv(pole_rd, pole_td, pole_nd)

    # genrating CSL
    axis_list = [np.array([1, 1, 1])]

    lattice_type = 'fcc'
    upper_limit = 3
    crystal_matrix = generate_crystal_matrix(lattice_type, upper_limit)

    df_csl = sigma_generate(crystal_matrix, axis_list)
    df_csl.to_csv('csl.csv')
    df_csl = pd.read_csv('csl.csv', index_col=None)

    # define a list of axes to be studied
    axis_list = [item / LA.norm(item) for item in axis_list]
    axis_list = [item.tolist() for item in axis_list]

    px = []
    py = []
    for ind, axis in enumerate(axis_list):
        px_temp, py_temp = stereographic_projection(pole_rd, pole_nd, axis)
        px.append(px_temp)
        py.append(py_temp)
    # ax1.scatter(px, py)

    # calculating dislocation density tensor, Scalar dislocation densities and lattice curvature
    # tilt boundary
    # creating lattice

    crystal_matrix1 = Crystal(lattice_type, upper_limit)  # upper limit for
    crystal_matrix2 = Crystal(lattice_type, upper_limit)  # upper limit for
    crystal_matrix2_shifted = Crystal(lattice_type, upper_limit)  # upper limit for

    # plot axis and two adjacent planes
    # chose an axis to generate layer

    # start from CSL, find DSC
    index_csl = 1

    axis_string = df_csl.loc[index_csl, 'uvw_list']
    axis = ast.literal_eval(axis_string)
    axis = axis / LA.norm(axis)
    angle = df_csl.loc[index_csl, 'angle'] * math.pi / 180
    print('csl information:')
    print(df_csl.iloc[index_csl])
    first_site_coordinates_string = df_csl.loc[index_csl, 'first_site']
    second_site_coordinates_string = df_csl.loc[index_csl, 'second_site']
    first_site_coordinates = ast.literal_eval(first_site_coordinates_string)
    second_site_coordinates = ast.literal_eval(second_site_coordinates_string)

    axis_plot = [[-axis[0], axis[0]], [-axis[1], axis[1]], [-axis[2], axis[2]]]
    plane1 = Plane(axis, crystal_matrix1.lattice, 0)  # (axis, lattice_objects, plane_index)
    plane2 = Plane(axis, crystal_matrix2.lattice, 0)
    plane2.rotate(axis, angle)

    # create dsc lattice using the two planes
    dsc = DSC(plane1, plane2)
    chossen_burgers_vector = dsc.burgers_vectors[2]
    print('chossen_burgers_vector=', chossen_burgers_vector)
    # shift plane 2 by the chossen_burgers_vector
    plane2_shifted = Plane(axis, crystal_matrix2_shifted.lattice, 0)
    plane2_shifted.rotate(axis, angle)

    chossen_burgers_vector_projected_on_plane2 = plane2_shifted.shift(chossen_burgers_vector)
    dsc_shifted = DSC(plane1, plane2_shifted)
    # scatter3D the whole crystal
    fig = plt.figure(1, figsize=(6, 6))
    ax1 = fig.add_subplot(111, projection='3d')
    for site in crystal_matrix1.lattice:
        if site.on_principle_axis:
            ax1.scatter3D(site.coordinates[0], site.coordinates[1], site.coordinates[2], marker='o', c='r', alpha=0.8)
        else:
            ax1.scatter3D(site.coordinates[0], site.coordinates[1], site.coordinates[2], marker='o', c='b', alpha=0.8)

    # using neighboring relation to plot the grid in 3D
    fig2 = plt.figure(2, figsize=(6, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    for site in dsc.lattice:
        if site.from_grain == 1:
            for neighbor in site.neighbors:
                ax2.plot3D([site.coordinates[0], neighbor.coordinates[0]],
                           [site.coordinates[1], neighbor.coordinates[1]],
                           [site.coordinates[2], neighbor.coordinates[2]], c='b')
        else:
            for neighbor in site.neighbors:
                ax2.plot3D([site.coordinates[0], neighbor.coordinates[0]],
                           [site.coordinates[1], neighbor.coordinates[1]],
                           [site.coordinates[2], neighbor.coordinates[2]], c='k')

    for site in plane2_shifted.lattice:
        ax2.scatter3D(site.coordinates[0], site.coordinates[1], site.coordinates[2], marker='o', c='r', alpha=0.8)
    arw = Arrow3D([0, chossen_burgers_vector[0]], [0, chossen_burgers_vector[1]], [0, chossen_burgers_vector[2]],
                  arrowstyle="->", color="purple", lw=1,
                  mutation_scale=10)
    # ax2.add_artist(arw)

    # using neighboring relation to plot the grid in 2D
    fig3 = plt.figure(3, figsize=(6, 6))
    ax3 = fig3.add_subplot(111)
    for site in plane1.lattice:
        for neighbor in site.neighbors:
            ax3.plot([site.x, neighbor.x],
                     [site.y, neighbor.y], c='grey',alpha = 0.6)
    for site in plane2.lattice:
        for neighbor in site.neighbors:
            ax3.plot([site.x, neighbor.x],
                     [site.y, neighbor.y], c='k',alpha=0.6)

    for site in plane2_shifted.lattice:
        ax3.scatter(site.x, site.y, marker='o', c='purple', alpha=0.4)
    for site in dsc.csl_sites:
        ax3.scatter(site.x, site.y, marker='o', c='r', alpha=1)
    for site in dsc_shifted.csl_sites:
        ax3.scatter(site.x, site.y, marker='s', c='b', alpha=1)
    # plot burgers vector
    ax3.plot([0, chossen_burgers_vector_projected_on_plane2[0]], [0, chossen_burgers_vector_projected_on_plane2[1]])


    # plot plane1 with each index labeled
    fig4 = plt.figure(4, figsize=(6, 6))
    ax4 = fig4.add_subplot(111)
    for site in plane1.lattice:
        if site.on_principle_axis:
            ax4.scatter(site.x, site.y, c='r')
        else:
            ax4.scatter(site.x, site.y, c='b')
        for neighbor in site.neighbors:
            ax4.plot([site.x, neighbor.x],
                     [site.y, neighbor.y], c='k')
    ind_xy_txt = []
    for site in plane1.lattice:
        ind_xy_txt.append(str(site.index_xy.tolist()))
    print(ind_xy_txt)
    for idx,site in enumerate(plane1.lattice):
        ax4.annotate(ind_xy_txt[idx], (site.x, site.y))
    ax4.plot([0, plane1.e1_planar[0]], [0, plane1.e1_planar[1]])
    ax4.plot([0, plane1.e2_planar[0]], [0, plane1.e2_planar[1]])

    plt.show()


if __name__ == "__main__":
    main()
