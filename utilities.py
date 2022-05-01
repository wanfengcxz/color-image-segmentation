from heapq import heappop, heappush
import random
from typing import List, Tuple
from PIL import Image
import numpy as np
from numba import jit, prange, njit
from sklearn import neighbors

from custom_types import ImageDimensions


@njit
def flattened_index(pixel_position: Tuple[int, int], img_dim: ImageDimensions) -> int:
    """ Get the index of a pixel in the flattened array """
    if pixel_position[0] < 0 or pixel_position[0] >= img_dim.rows:
        raise IndexError("Row index out of bounds")
    if pixel_position[1] < 0 or pixel_position[1] >= img_dim.cols:
        raise IndexError("Column index out of bounds")
    return pixel_position[0] * img_dim.cols + pixel_position[1]


@njit
def expanded_index(flattened_index: int, img_dim: ImageDimensions) -> Tuple[int, int]:
    """ Get the row and column of a pixel in the expanded array """
    if flattened_index >= img_dim.rows * img_dim.cols:
        raise IndexError("Index out of bounds")
    return (flattened_index // img_dim.cols, flattened_index % img_dim.cols)


def is_border_pixel(pixel_position: Tuple[int, int], img_dim: ImageDimensions) -> bool:
    """ Check if a pixel is on the border of the image """
    if pixel_position[0] == 0 or pixel_position[1] == 0:
        return True
    if pixel_position[0] == img_dim.rows-1 or pixel_position[1] == img_dim.cols-1:
        return True
    else:
        return False


@njit
def is_segment_border_pixel(pixel_position: Tuple[int, int], img_dim, segments) -> bool:
    """ Check if a pixel is on the border of a segment """
    pixel_index = flattened_index(pixel_position, img_dim)
    for y in range(-1, 2):
        for x in range(-1, 2):
            pos = (pixel_position[0]+y, pixel_position[1]+x)
            try:
                neighbour_index = flattened_index(pos, img_dim)
                if segments[neighbour_index] != segments[pixel_index]:
                    return True
            except Exception:
                continue
    return False


def save_segmented_img(filepath: str, img_data, segments, type: int = 2):
    if type not in [1, 2]:
        raise ValueError("Type must be 1 or 2")

    new_img_data = img_data.copy()
    img_dimensions = ImageDimensions(img_data.shape[0], img_data.shape[1])

    # Check every pixel if it is a border pixel
    for y in range(img_dimensions.rows):
        for x in range(img_dimensions.cols):
            pixel_pos = (y, x)

            if type == 1:  # Type 1 segmentation
                if (is_border_pixel(pixel_pos, img_dimensions) or is_segment_border_pixel(pixel_pos, img_dimensions, segments)):
                    # Make border pixels green
                    new_img_data[y, x, 0] = 0
                    new_img_data[y, x, 1] = 255
                    new_img_data[y, x, 2] = 0

            elif type == 2:  # Type 2 segmentation
                if (is_border_pixel(pixel_pos, img_dimensions) or is_segment_border_pixel(pixel_pos, img_dimensions, segments)):
                    # Make border pixels black
                    new_img_data[y, x, 0] = 0
                    new_img_data[y, x, 1] = 0
                    new_img_data[y, x, 2] = 0
                else:
                    # And everything else white
                    new_img_data[y, x, 0] = 255
                    new_img_data[y, x, 1] = 255
                    new_img_data[y, x, 2] = 255

    new_img_data = new_img_data.astype(np.int8)
    Image.fromarray(new_img_data, "RGB").save(
        filepath+"type_"+str(type)+".jpg")


@njit
def get_flattened_neighbours(initial_index: int, img_dim: ImageDimensions, use_diagonals: bool = False) -> List[int]:
    """ Get the indices of the neighbours of a pixel """
    if initial_index >= img_dim.rows * img_dim.cols:
        raise IndexError("Index out of bounds")
    neighbors = []
    # If not on first row, add the pixel above
    if initial_index - img_dim.cols >= 0:
        neighbors.append(initial_index - img_dim.cols)

        if use_diagonals:
            # If not on the first column add the pixel up and to the left
            if initial_index % img_dim.cols != 0:
                neighbors.append(initial_index - img_dim.cols - 1)

            # If not on the last column add the pixel up and to the right
            if initial_index % img_dim.cols != img_dim.cols-1:
                neighbors.append(initial_index - img_dim.cols + 1)

    # If not on last row, add the pixel below
    if initial_index + img_dim.cols < img_dim.rows * img_dim.cols:
        neighbors.append(initial_index + img_dim.cols)

        if use_diagonals:
            # If not on the first column add the pixel down and to the left
            if initial_index % img_dim.cols != 0:
                neighbors.append(initial_index + img_dim.cols - 1)

            # If not on the last column add the pixel down and to the right
            if initial_index % img_dim.cols != img_dim.cols-1:
                neighbors.append(initial_index + img_dim.cols + 1)

    # If not on first column, add the pixel to the left
    if initial_index % img_dim.cols != 0:
        neighbors.append(initial_index - 1)
    # If not on last column, add the pixel to the right
    if initial_index % img_dim.cols != img_dim.cols - 1:
        neighbors.append(initial_index + 1)
    return neighbors


@njit
def euclidean_distance(p1_rgb: Tuple[int, int, int], p2_rgb: Tuple[int, int, int]) -> float:
    """ Calculate the euclidean distance in RGB color space between the two pixels """
    r1, b1, g1 = p1_rgb
    r2, b2, g2 = p2_rgb

    return np.sqrt((r1-r2)**2 + (b1-b2)**2 + (g1-g2)**2)


# @njit
# def prims_algorithm(img_data):
#     """ Create a minimal spanning tree using prim's algorithm """
#     img_dim = ImageDimensions(img_data.shape[0], img_data.shape[1])

#     # Initialise empty array of lenght num_rows*num_cols
#     genotype = np.zeros(img_dim.rows*img_dim.cols, dtype=np.int32)

#     visited = set()

#     # Select a random pixel as the first pixel
#     node = np.random.randint(0, len(genotype))
#     visited.add(node)

#     neighbors = get_flattened_neighbours(node, img_dim)
#     potential_edges = []
#     used_edges = []

#     while len(img_data) > len(visited):
#         # Get the neighbours of the current node
#         neighbors = get_flattened_neighbours(node, img_dim)

#         # Remove neighbours already visited
#         for neighbor in neighbors:
#             if neighbor in visited:
#                 neighbors.remove(neighbor)

#         # Calculate the distance between the current node and the neighbours
#         for neighbor in neighbors:
#             rgb_distance = euclidean_distance(
#                 img_data[expanded_index(node, img_dim)], img_data[expanded_index(neighbor, img_dim)])
#             potential_edges.append((rgb_distance, node, neighbor))

#         # Sort the potential edges by distance
#         potential_edges.sort(key=lambda x: x[0])

#         # Get the shortest edge until its destination is not in the visited set
#         while potential_edges[0][2] in visited:
#             potential_edges.pop(0)

#         # Add the edge to the genotype
#         genotype[potential_edges[0][1]] = potential_edges[0][2]

#         # Add the destination to the visited set
#         visited.add(potential_edges[0][2])

#         # Set the current node to the destination
#         node = potential_edges[0][2]

#         # Remove the edge from the potential edges
#         used_edges.append(potential_edges.pop(0))

#     used_edges.sort(reverse=True)
#     for i in range(6):
#         v_to = used_edges[i][2]
#         genotype[v_to] = v_to
#     return genotype

@njit
def prims_algorithm(img_data):
    img_dim = ImageDimensions(img_data.shape[0], img_data.shape[1])
    genotype = np.zeros(img_dim.rows*img_dim.cols, dtype=np.int32)
    visited_nodes = set()

    current_node = random.randint(0, len(genotype)-1)

    potential_edges = [(0.0, 0, 0) for x in range(0)]

    while len(visited_nodes) < len(genotype):
        if current_node not in visited_nodes:
            visited_nodes.add(current_node)
            neighbours = get_flattened_neighbours(current_node, img_dim)
            for neighbour in neighbours:
                node_rgb = img_data[expanded_index(current_node, img_dim)]
                destination_rgb = img_data[expanded_index(neighbour, img_dim)]
                rgb_distance = euclidean_distance(node_rgb, destination_rgb)
                heappush(potential_edges,
                         (rgb_distance, current_node, neighbour))

        _, starting_index, destination_index = heappop(potential_edges)

        while destination_index in visited_nodes and len(potential_edges) > 0:
            _, starting_index, destination_index = heappop(potential_edges)

        genotype[destination_index] = starting_index

        current_node = destination_index
    return genotype


def separate_segments(genotype) -> np.ndarray:
    """ Separate the image into segments by traversing the genotype """

    # List of the ids of each pixels segment
    segment_ids = [None for _ in range(len(genotype))]
    # segment_ids = np.full(len(genotype), None, dtype=np.int32)
    current_segment_id = 0
    for i in range(len(genotype)):
        # If already included in a segment, then go to next pixel
        if segment_ids[i] != None:
            continue
        current_segment = []
        current_segment.append(i)
        segment_ids[i] = current_segment_id
        next = genotype[i]

        # Traverse until we reach a pixel already included in a segment
        while segment_ids[next] == None:
            segment_ids[next] = current_segment_id
            current_segment.append(next)
            next = genotype[next]

        # When hitting a pixel already in the same segment, go to next segment
        if segment_ids[next] == current_segment_id:
            current_segment_id += 1
            continue
        # Else, combine the two segments
        else:
            new_segment_id = segment_ids[next]
            for j in current_segment:
                # Update the segment id of the current segment to new (old) segment id
                segment_ids[j] = new_segment_id

    # Needs to be numpy array for Numba to compile
    return np.array(segment_ids)


@njit
def calculate_edge_value(img_data, segment_ids) -> float:
    """ Calculate the edge value for all segments in the image. """
    img_dim = ImageDimensions(img_data.shape[0], img_data.shape[1])
    edge_value = 0
    for i in range(len(segment_ids)):
        neighbours = get_flattened_neighbours(i, img_dim)
        for neighbour in neighbours:
            # If in same segment
            if segment_ids[i] == segment_ids[neighbour]:
                continue
            # Neighbour is in different segment
            node_rgb = img_data[expanded_index(i, img_dim)]
            neighbor_rgb = img_data[expanded_index(neighbour, img_dim)]
            rgb_distance = euclidean_distance(node_rgb, neighbor_rgb)
            edge_value += rgb_distance

    return edge_value


@njit
def calculate_connectivity(segment_ids, img_dim: ImageDimensions) -> float:
    """ Calculate the connecitvity measure, evaluates the degree to which neighboring pixels have been placed in the same segment  """
    connectivity = 0

    for i in range(len(segment_ids)):
        # Get all neighbors of a pixel if neighbor is of a different segment
        neighbours = get_flattened_neighbours(i, img_dim)
        for neighbour in neighbours:
            # If not in the same segment
            if segment_ids[i] != segment_ids[neighbour]:
                connectivity += (1/8)

    return connectivity


# @jit(parallel=True, nopython=True)
# def mean_numba(a):

#     res = []
#     for i in prange(a.shape[0]):
#         res.append(a[i, :].mean())

#     return np.array(res)


@njit
def calculate_deviation(img_data, segment_ids) -> float:
    """ Calculate the deviation, a measure of similiarity of pixels in the same segment """
    img_dim = ImageDimensions(img_data.shape[0], img_data.shape[1])
    # Calculate centroids of each segment
    centroids = []
    # Iterate all segments
    for segment_index in range(max(segment_ids)+1):
        r = []
        g = []
        b = []
        for j in range(len(segment_ids)):
            if segment_ids[j] == segment_index:
                r.append(img_data[expanded_index(j, img_dim)][0])
                g.append(img_data[expanded_index(j, img_dim)][1])
                b.append(img_data[expanded_index(j, img_dim)][2])
        r_mean = sum(r) / len(r)
        g_mean = sum(g) / len(g)
        b_mean = sum(b) / len(b)
        centroids.append((r_mean, g_mean, b_mean))

    # Calculate deviation of each pixel from its centroid
    deviation = 0.0
    for segment_index in range(len(segment_ids)):
        node_rgb = img_data[expanded_index(segment_index, img_dim)]
        centroid = centroids[segment_ids[segment_index]]
        deviation += euclidean_distance(node_rgb, centroid)

    return deviation
