#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
from scipy import ndimage as ndi

cimport numpy as cnp
from _ccomp cimport find_root, join_trees

from skimage.util import img_as_float64
from skimage._shared.utils import warn

cnp.import_array()


def felzenszwalb_cython_3d(image, double scale=1, sigma=0.8, Py_ssize_t min_size=20, spacing=(1,1,1)):
    """
    Code modified from: https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb

    Felzenszwalb's efficient graph based segmentation for
    single or multiple channels.

    Produces an oversegmentation of a single or multi-channel image
    using a fast, minimum spanning tree based clustering on the image grid.
    The number of produced segments as well as their size can only be
    controlled indirectly through ``scale``. Segment size within an image can
    vary greatly depending on local contrast.

    Parameters
    ----------
    image : (N, M, C) ndarray
        Input image.
    scale : float, optional (default 1)
        Sets the obervation level. Higher means larger clusters.
    sigma : float, optional (default 0.8)
        Width of Gaussian smoothing kernel used in preprocessing.
        Larger sigma gives smother segment boundaries.
    min_size : int, optional (default 20)
        Minimum component size. Enforced using postprocessing.

    Returns
    -------
    segment_mask : (N, M) ndarray
        Integer mask indicating segment labels.
    """


    image = img_as_float64(image)
    dtype = image.dtype

    # rescale scale to behave like in reference implementation
    scale = float(scale) / 255.

    spacing = np.ascontiguousarray(spacing, dtype=dtype)
    sigma = np.array([sigma, sigma, sigma], dtype=dtype)
    sigma /= spacing.astype(dtype)

    image = ndi.gaussian_filter(image, sigma=sigma)
    height, width, depth = image.shape  # depth, height, width!
    image = image[..., None]

    # assuming spacing is equal in xy dir.
    s = spacing[0]/spacing[1]
    w1 = 1.0  # x, y, xy
    w2 = s**2  # z
    w3 = (np.sqrt(1+s**2)/np.sqrt(2))**2  # zx, zy
    w4 = (np.sqrt(2 + s**2)/np.sqrt(3))**2  # zxy


    cost1 = np.sqrt(w1 * np.sum((image[:, 1:, :] - image[:, :width-1, :])**2, axis=-1))  # x
    cost2 = np.sqrt(w1 * np.sum((image[:, 1:, 1:] - image[:, :width-1, :depth-1])**2, axis=-1)) # xy
    cost3 = np.sqrt(w1 * np.sum((image[:, :, 1:] - image[:, :, :depth-1])**2, axis=-1)) # y
    cost7 = np.sqrt(w1 * np.sum((image[:, 1:, :depth-1] - image[:, :width-1, 1:])**2, axis=-1)) # xy
    cost9 = np.sqrt(w3 * np.sum((image[1:, 1:, :] - image[:height-1, :width-1, :])**2, axis=-1)) # zx
    cost10 = np.sqrt(w4 * np.sum((image[1:, 1:, 1:] - image[:height-1, :width-1, :depth-1])**2, axis=-1)) # zxy
    cost11 = np.sqrt(w3 * np.sum((image[1:, :, 1:] - image[:height-1, :, :depth-1])**2, axis=-1)) # zy
    cost12 = np.sqrt(w3 * np.sum((image[1:, :width-1, :] - image[:height-1, 1:, :])**2, axis=-1)) # zx
    cost13 = np.sqrt(w4 * np.sum((image[1:, :width-1, :depth-1] - image[:height-1, 1:, 1:])**2, axis=-1)) # zxy
    cost14 = np.sqrt(w3 * np.sum((image[1:, :, :depth-1] - image[:height-1, :, 1:])**2, axis=-1)) # zy
    cost15 = np.sqrt(w4 * np.sum((image[1:, 1:, :depth-1] - image[:height-1, :width-1, 1:])**2, axis=-1)) # zxy
    cost16 = np.sqrt(w4 * np.sum((image[1:, :width-1, 1:] - image[:height-1, 1:, :depth-1])**2, axis=-1)) # zxy
    cost25 = np.sqrt(w2 * np.sum((image[1:, :, :] - image[:height-1, :, :])**2, axis=-1)) # z


    cdef cnp.ndarray[cnp.float_t, ndim=1] costs = np.hstack([cost1.ravel(), cost2.ravel(), cost3.ravel(), cost7.ravel(), cost9.ravel(), cost10.ravel(), cost11.ravel(), cost12.ravel(), cost13.ravel(), cost14.ravel(), cost15.ravel(), cost16.ravel(), cost25.ravel()]).astype(float)

    # compute edges between pixels:
    cdef cnp.ndarray[cnp.intp_t, ndim=3] segments \
            = np.arange(width * height * depth, dtype=np.intp).reshape(height, width, depth)


    edges1 = np.c_[segments[:, 1:, :].ravel(), segments[:, :width-1, :].ravel()]
    edges2 = np.c_[segments[:, 1:, 1:].ravel(), segments[:, :width-1, :depth-1].ravel()]
    edges3 = np.c_[segments[:, :, 1:].ravel(), segments[:, :, :depth-1].ravel()]
    edges7 = np.c_[segments[:, 1:, :depth-1].ravel(), segments[:, :width-1, 1:].ravel()]
    edges9 = np.c_[segments[1:, 1:, :].ravel(), segments[:height-1, :width-1, :].ravel()]
    edges10 = np.c_[segments[1:, 1:, 1:].ravel(), segments[:height-1, :width-1, :depth-1].ravel()]
    edges11 = np.c_[segments[1:, :, 1:].ravel(), segments[:height-1, :, :depth-1].ravel()]
    edges12 = np.c_[segments[1:, :width-1, :].ravel(), segments[:height-1, 1:, :].ravel()]
    edges13 = np.c_[segments[1:, :width-1, :depth-1].ravel(), segments[:height-1, 1:, 1:].ravel()]
    edges14 = np.c_[segments[1:, :, :depth-1].ravel(), segments[:height-1, :, 1:].ravel()]
    edges15 = np.c_[segments[1:, 1:, :depth-1].ravel(), segments[:height-1, :width-1, 1:].ravel()]
    edges16 = np.c_[segments[1:, :width-1, 1:].ravel(), segments[:height-1, 1:, :depth-1].ravel()]
    edges25 = np.c_[segments[1:, :, :].ravel(), segments[:height-1, :, :].ravel()]

    cdef cnp.ndarray[cnp.intp_t, ndim=2] edges \
            = np.vstack([edges1, edges2, edges3, edges7, edges9, edges10, edges11, edges12, edges13, edges14, edges15, edges16, edges25])

    # initialize data structures for segment size
    # and inner cost, then start greedy iteration over edges.
    edge_queue = np.argsort(costs)
    edges = np.ascontiguousarray(edges[edge_queue])
    costs = np.ascontiguousarray(costs[edge_queue])
    cdef cnp.intp_t *segments_p = <cnp.intp_t*>segments.data
    cdef cnp.intp_t *edges_p = <cnp.intp_t*>edges.data
    cdef cnp.float_t *costs_p = <cnp.float_t*>costs.data
    cdef cnp.ndarray[cnp.intp_t, ndim=1] segment_size \
            = np.ones(width * height * depth, dtype=np.intp)

    # inner cost of segments
    cdef cnp.ndarray[cnp.float_t, ndim=1] cint = np.zeros(width * height * depth)
    cdef cnp.intp_t seg0, seg1, seg_new, e
    cdef float cost, inner_cost0, inner_cost1
    cdef Py_ssize_t num_costs = costs.size

    with nogil:
        # set costs_p back one. we increase it before we use it
        # since we might continue before that.
        costs_p -= 1
        for e in range(num_costs):
            seg0 = find_root(segments_p, edges_p[0])
            seg1 = find_root(segments_p, edges_p[1])

            edges_p += 2
            costs_p += 1
            if seg0 == seg1:
                continue


            inner_cost0 = cint[seg0] + scale / segment_size[seg0]
            inner_cost1 = cint[seg1] + scale / segment_size[seg1]

#    return 0 # ok

            if costs_p[0] < min(inner_cost0, inner_cost1):
                # update size and cost

                join_trees(segments_p, seg0, seg1)  # TODO: not ok!
    #return 0  # not ok!!
                seg_new = find_root(segments_p, seg0)
                segment_size[seg_new] = segment_size[seg0] + segment_size[seg1]
                cint[seg_new] = costs_p[0]


        # postprocessing to remove small segments
        edges_p = <cnp.intp_t*>edges.data
        for e in range(num_costs):
            seg0 = find_root(segments_p, edges_p[0])
            seg1 = find_root(segments_p, edges_p[1])
            edges_p += 2
            if seg0 == seg1:
                continue
            if segment_size[seg0] < min_size or segment_size[seg1] < min_size:
                join_trees(segments_p, seg0, seg1)
                seg_new = find_root(segments_p, seg0)
                segment_size[seg_new] = segment_size[seg0] + segment_size[seg1]



    # unravel the union find tree
    flat = segments.ravel()
    old = np.zeros_like(flat)
    while (old != flat).any():
        old = flat
        flat = flat[flat]
    flat = np.unique(flat, return_inverse=True)[1]
    return flat.reshape((height, width, depth))