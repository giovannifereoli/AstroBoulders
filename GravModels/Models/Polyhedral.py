import os
import numpy as np
import trimesh
from numba import njit, prange


@njit(cache=True)
def calculate_edge_dyads(
    vertices, faces, edges_unique, face_adjacency_edges, face_normals, face_adjacency
):
    """
    Calculate the edge dyads for a given set of vertices, faces, and edge information.

    Parameters:
    vertices (numpy.ndarray): Array of vertex coordinates.
    faces (numpy.ndarray): Array of face indices.
    edges_unique (numpy.ndarray): Array of unique edge indices.
    face_adjacency_edges (numpy.ndarray): Array of face adjacency edge indices.
    face_normals (numpy.ndarray): Array of face normals.
    face_adjacency (numpy.ndarray): Array of face adjacency information.

    Returns:
    numpy.ndarray: Array of edge dyads.

    """
    edge_dyads = np.zeros((len(edges_unique), 3, 3))  # In order of unique edges
    for i in prange(len(edges_unique)):

        P1 = vertices[edges_unique[i][0]]
        P2 = vertices[edges_unique[i][1]]

        facet_A_idx = int(face_adjacency[i][0])
        facet_B_idx = int(face_adjacency[i][1])
        normal_A = face_normals[facet_A_idx]
        normal_B = face_normals[facet_B_idx]

        face_A_vertices = faces[facet_A_idx]
        face_B_vertices = faces[facet_B_idx]
        face_A_c = (
            vertices[face_A_vertices[0]]
            + vertices[face_A_vertices[1]]
            + vertices[face_A_vertices[2]]
        ) / 3.0
        face_B_c = (
            vertices[face_B_vertices[0]]
            + vertices[face_B_vertices[1]]
            + vertices[face_B_vertices[2]]
        ) / 3.0

        B_2_A = face_A_c - face_B_c
        A_2_B = face_B_c - face_A_c

        edge_direction = P1 - P2
        edge_direction /= np.linalg.norm(edge_direction)

        edge_normal_A_to_B = np.cross(normal_A, edge_direction)
        edge_normal_B_to_A = np.cross(normal_B, edge_direction)

        if np.dot(A_2_B, edge_normal_A_to_B) < 0:
            edge_normal_A_to_B *= -1.0
        if np.dot(B_2_A, edge_normal_B_to_A) < 0:
            edge_normal_B_to_A *= -1.0

        dyad_A = np.outer(normal_A, edge_normal_A_to_B)
        dyad_B = np.outer(normal_B, edge_normal_B_to_A)

        edge_dyads[i] = dyad_A + dyad_B
    return edge_dyads


@njit(cache=True)
def calculate_facet_dyads(face_normals):
    """
    Calculate the dyads of the facet normals.

    Args:
        face_normals (numpy.ndarray): Array of face normals.

    Returns:
        numpy.ndarray: Array of dyads of the facet normals.
    """
    facet_dyads = np.zeros((len(face_normals), 3, 3))
    for i in prange(len(face_normals)):
        facet_normal = face_normals[i]
        facet_dyads[i] = np.outer(facet_normal, facet_normal)
    return facet_dyads


@njit(cache=True)
def GetPerformanceFactor(r_scaled, vertices, faces, facet_idx):
    """
    Calculates the performance factor for a given set of parameters.

    Parameters:
    - r_scaled: The scaled position vector.
    - vertices: The list of vertices.
    - faces: The list of faces.
    - facet_idx: The index of the facet.

    Returns:
    - The performance factor.

    """
    r0 = vertices[int(faces[facet_idx][0])]
    r1 = vertices[int(faces[facet_idx][1])]
    r2 = vertices[int(faces[facet_idx][2])]

    r0m = r0 - r_scaled
    r1m = r1 - r_scaled
    r2m = r2 - r_scaled

    R0 = np.linalg.norm(r0m)
    R1 = np.linalg.norm(r1m)
    R2 = np.linalg.norm(r2m)

    r1m_cross_r2m = np.cross(r1m, r2m)

    return 2.0 * np.arctan2(
        np.dot(r0m, r1m_cross_r2m),
        R0 * R1 * R2
        + R0 * np.dot(r1m, r2m)
        + R1 * np.dot(r0m, r2m)
        + R2 * np.dot(r0m, r1m),
    )


@njit(cache=True)
def GetLe(r_scaled, vertices, edges_unique, edge_idx):
    """
    Calculate the logarithm of the ratio of the sum of distances between three points to the difference of the sum of distances between two points.

    Parameters:
    r_scaled (numpy.ndarray): The scaled position vector.
    vertices (numpy.ndarray): The array of vertices.
    edges_unique (numpy.ndarray): The array of unique edges.
    edge_idx (int): The index of the edge.

    Returns:
    float: The logarithm of the ratio of the sum of distances to the difference of the sum of distances.

    """
    r0 = vertices[int(edges_unique[edge_idx][0])]
    r1 = vertices[int(edges_unique[edge_idx][1])]

    r0m = r0 - r_scaled
    r1m = r1 - r_scaled
    rem = r1m - r0m

    R0 = np.linalg.norm(r0m)
    R1 = np.linalg.norm(r1m)
    Re = np.linalg.norm(rem)

    return np.log((R0 + R1 + Re) / (R0 + R1 - Re))


@njit(cache=True)
def facet_acc_loop(point_scaled, vertices, faces, facet_dyads):
    """
    Calculate the acceleration, potential, and Laplacian for a given point in a polyhedral model.

    Parameters:
    - point_scaled (numpy.ndarray): The scaled coordinates of the point.
    - vertices (numpy.ndarray): The vertices of the polyhedral model.
    - faces (numpy.ndarray): The faces of the polyhedral model.
    - facet_dyads (numpy.ndarray): The dyads associated with each face of the polyhedral model.

    Returns:
    - acc (numpy.ndarray): The acceleration vector.
    - pot (float): The potential energy.
    - lap (float): The Laplacian.

    """
    acc = np.zeros((3,))
    pot = 0.0
    lap = 0.0
    for i in prange(len(faces)):
        r0 = vertices[faces[i][0]]

        r_e = r0 - point_scaled
        r_f = r_e  # Page 11

        wf = GetPerformanceFactor(point_scaled, vertices, faces, i)
        F = facet_dyads[i]

        acc += wf * np.dot(F, r_f)
        pot -= wf * np.dot(r_f, np.dot(F, r_f))
        lap += wf
    return acc, pot, lap


@njit(cache=True)
def edge_acc_loop(point_scaled, vertices, edges_unique, edge_dyads):
    """
    Calculate the acceleration and potential energy contribution for a given point due to the edges of a polyhedral model.

    Parameters:
    - point_scaled: numpy array representing the scaled coordinates of the point
    - vertices: numpy array representing the vertices of the polyhedral model
    - edges_unique: list of tuples representing the unique edges of the polyhedral model
    - edge_dyads: list of numpy arrays representing the dyads of the edges of the polyhedral model

    Returns:
    - acc: numpy array representing the acceleration vector
    - pot: float representing the potential energy contribution

    """
    acc = np.zeros((3,))
    pot = 0.0
    for i in prange(len(edges_unique)):
        r0 = vertices[edges_unique[i][0]]
        r1 = vertices[edges_unique[i][1]]

        r_middle = (r0 + r1) / 2

        r_e = r_middle - point_scaled

        Le = GetLe(point_scaled, vertices, edges_unique, i)
        E = edge_dyads[i]

        acc -= Le * np.dot(E, r_e)
        pot += Le * np.dot(r_e, np.dot(E, r_e))

    return acc, pot


class Polyhedral:
    """
    Class for computing gravitational acceleration, potential, and Laplacian
    for a polyhedral shape model of a celestial body.

    Attributes:
        density (float): Density of the celestial body.
        scaleFactor (float): Scaling factor for converting mesh units (e.g., km to meters).
        mesh (trimesh.Trimesh): Triangular mesh representation of the celestial body.
        facet_dyads (numpy.ndarray): Dyads calculated for each face's normal vector.
        edge_dyads (numpy.ndarray): Dyads calculated for each edge's normal vector.
    """

    def __init__(self, celestial_body):
        """
        Initializes PolyhedralModel with a celestial body.

        Args:
            celestial_body (CelestialBody): Body from which gravity measurements are produced.
        """
        self.planet = celestial_body
        self.density = self.planet.density
        self.G = 6.67408 * 1e-11
        self.scaleFactor = 1e3  # Assume that the mesh is given in km
        self.obj_file = celestial_body.model
        _, file_extension = os.path.splitext(self.obj_file)
        self.mesh = trimesh.load_mesh(self.obj_file, file_type=file_extension[1:])

        self.facet_dyads = calculate_facet_dyads(self.mesh.face_normals)
        self.edge_dyads = calculate_edge_dyads(
            self.mesh.vertices,
            self.mesh.faces,
            self.mesh.edges_unique,
            self.mesh.face_adjacency_edges,
            self.mesh.face_normals,
            self.mesh.face_adjacency,
        )

    def calculate_acceleration(self, positions):
        """
        calculate gravitational acceleration at given positions.

        Args:
            positions (numpy.ndarray): Array of position vectors where acceleration is calculated.

        Returns:
            numpy.ndarray: Array of gravitational accelerations at the given positions.
        """
        point_scaled = positions / self.scaleFactor
        accelerations = np.zeros_like(positions)

        for i in range(len(positions)):
            acc_facet, _, _ = facet_acc_loop(
                point_scaled[i], self.mesh.vertices, self.mesh.faces, self.facet_dyads
            )
            acc_edge, _ = edge_acc_loop(
                point_scaled[i],
                self.mesh.vertices,
                self.mesh.edges_unique,
                self.edge_dyads,
            )

            acc = acc_facet + acc_edge
            acc *= self.G * self.density * self.scaleFactor
            accelerations[i] = acc

        return accelerations

    def calculate_potential(self, positions):
        """
        calculate gravitational potential at given positions.

        Args:
            positions (numpy.ndarray): Array of position vectors where potential is calculated.

        Returns:
            numpy.ndarray: Array of gravitational potentials at the given positions.
        """
        point_scaled = positions / self.scaleFactor
        potentials = np.zeros(len(positions))

        for i in range(len(positions)):
            _, pot_facet, _ = facet_acc_loop(
                point_scaled[i], self.mesh.vertices, self.mesh.faces, self.facet_dyads
            )
            pot_edge, _ = edge_acc_loop(
                point_scaled[i],
                self.mesh.vertices,
                self.mesh.edges_unique,
                self.edge_dyads,
            )

            pot = pot_edge + pot_facet
            pot *= (
                1.0 / 2.0 * self.G * self.density * self.scaleFactor**2
            )  # [km^2/s^2] -> [m^2/s^2]

            potentials[i] = pot

        return potentials
