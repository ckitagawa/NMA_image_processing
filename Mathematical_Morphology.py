### From Prof. Abukhdeir's python file about nanowires
from numpy import *
from scipy import ndimage
import cv2
from matplotlib import pyplot

#class Mathematical_Morphology(object):
 #   def __init__(self, image):
  #      self.image = image
    ################################################################################
    # Mathematical Morphology Functions

def _hit_and_miss(image, hit, miss):
    "Implementation of the hit-or-miss transform"

    res1 = _erosion(image, hit)
    res2 = _erosion(image, miss, invert = True)
    
    return logical_and(res1, res2)

def _thin(image, hit, miss):
    "Implementation of morphological thinning"
    
    hm = _hit_and_miss(image, hit, miss)
    res = logical_and(image, logical_not(hm))

    return where(res, 1, 0)

def _erosion(image, kernel, invert = False):
    "Implementation of morphological erosion"

    image = where(image, 1, 0) 
    thresh = kernel.sum()
    
    result = ndimage.convolve(image, kernel, mode="constant")

    if invert == False:
        return where(result == thresh, 1, 0) 
    else:
        return where(result == 0, 1, 0)

def skeletonize_image(bw_image, prunings=0, display=False):
    """Skeletonize and prune a binary image through the application of hit-or-miss
     transforms using appropriate structure elements."""

    # generate the skeleton through thinning using the structure elements from:
    # http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm
    hit1 = array([[0, 0, 0],
                  [0, 1, 0],
                  [1, 1, 1]], dtype=uint8)
    miss1 = array([[1, 1, 1],
                   [0, 0, 0],
                   [0, 0, 0]], dtype=uint8)
    hit2 = array([[0, 0, 0],
                  [1, 1, 0],
                  [0, 1, 0]], dtype=uint8)
    miss2 = array([[0, 1, 1],
                   [0, 0, 1],
                   [0, 0, 0]], dtype=uint8)
    
    # create the initial list of hit and miss structure elements
    hit_list = [hit1, hit2]
    miss_list = [miss1, miss2]
    
    # use some nifty NumPy slicing to add the three remaining rotations of each
    # of the structure elements to the hit and miss lists
    for ii in range(6):
        hit_list.append(transpose(hit_list[-2])[::-1, ...])
        miss_list.append(transpose(miss_list[-2])[::-1, ...])

    # create a deep copy of the binary image for performing the skeletonization
    image = bw_image.copy()
    while 1:
        last = image
        for hit, miss in zip(hit_list, miss_list):
            hm = _hit_and_miss(image, hit, miss)
            image = logical_and(image, logical_not(hm))
        if logical_xor(last, image).sum() == 0:
            break
    
    # some of the structure elements might have responding more than once, enforce
    # that the skeleton is also a binary image
    skel_image = where(image, 1, 0)

    # prune end-points if the user requests it, pruning helps to remove artifacts 
    # resulting from the skeletonization
    
    # again, using the pruning structure elements from:
    # http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm
    hit1 = array([[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]])
    miss1 = array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 0, 0]])
    hit2 = array([[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]])
    miss2 = array([[1, 1, 1],
                  [1, 0, 1],
                  [0, 0, 1]])
    hit_list = [hit1, hit2]
    miss_list = [miss1, miss2]
    for ii in range(6):
        hit_list.append(transpose(hit_list[-2])[::-1, ...])
        miss_list.append(transpose(miss_list[-2])[::-1, ...])
    
    # first thin the endpoints twice to prune the artifacts from skeletonization
    pruned_skel_image = skel_image.copy()
    for i in range(prunings):
        for hit, miss in zip(hit_list, miss_list):
            pruned_skel_image = _thin(pruned_skel_image, hit, miss)


    if display != False:
        fig1 = pyplot.figure()
        pyplot.imshow(skel_image, cmap=pyplot.get_cmap("gray"))
        fig2 = pyplot.figure()
        pyplot.imshow(pruned_skel_image, cmap=pyplot.get_cmap("gray"))
        
        if display == True:
            pyplot.show()
        else:
            fig1.savefig(display + "_raw_skeleton.png")
            fig2.savefig(display + "_pruned_skeleton.png")    
        
    return(pruned_skel_image)


def find_branch_points(bw_image, skel_image, display=False):
    """
    Starting with a morphological skeleton, creates a corresponding binary image
    with all branch-points pixels (1) and all other pixels (0).
    """

    # identify 3-way branch-points through convolving the image using appropriate 
    # structure elements for an 8-connected skeleton:
    # http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm
    hit1 = array([[0, 1, 0],
                  [0, 1, 0],
                  [1, 0, 1]], dtype=uint8)
    hit2 = array([[1, 0, 0],
                  [0, 1, 0],
                  [1, 0, 1]], dtype=uint8)
    hit3 = array([[1, 0, 0],
                   [0, 1, 1],
                   [0, 1, 0]], dtype=uint8)
    hit_list = [hit1, hit2, hit3]

    # use some nifty NumPy slicing to add the three remaining rotations of each
    # of the structure elements to the hit list
    for ii in range(9):
        hit_list.append(transpose(hit_list[-3])[::-1, ...])

    # add structure elements for branch-points four 4-way branchpoints, these
    hit3 = array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]], dtype=uint8)
    hit4 = array([[1, 0, 1],
                  [0, 1, 0],
                  [1, 0, 1]], dtype=uint8)
    hit_list.append(hit3)
    hit_list.append(hit4)
    
    # create a zero array of the same shape as the skeleton and use it to collect
    # "hits" from the convolution operation
    branch_points = zeros(bw_image.shape)
    for hit in hit_list:
        target = hit.sum()
        curr = ndimage.convolve(skel_image, hit, mode="constant")
        branch_points = logical_or(branch_points, where(curr == target, 1, 0) )

    # pixels may "hit" multiple structure elements, ensure the output is a binary
    # image
    branch_points_image = where(branch_points, 1, 0)

    # use SciPy's ndimage module to label each contiguous foreground feature 
    # uniquely, this will locating and determining coordinates of each branch-point
    (labels, num_labels) = ndimage.label(branch_points_image)
    
    # use SciPy's ndimage module to determine the coordinates/pixel corresponding
    # to the center of mass of each branchpoint
    branch_points = ndimage.center_of_mass(bw_image, labels=labels, index=range(1, num_labels+1))
    branch_points = array([value for value in branch_points if not isnan(value[0]) or not isnan(value[1])], dtype=int)
    num_branch_points = len(branch_points)

    if display== True:
        pyplot.figure()
        pyplot.imshow(bw_image, cmap=pyplot.get_cmap("gray"))
        pyplot.scatter(branch_points[:,1], branch_points[:,0], marker='.', c="b", s=64, lw=0)
        pyplot.show()
    
    # return a list of branch-point coordinates (x, y) and the number of branch-
    # points
    return(branch_points, num_branch_points)


def find_end_points(bw_image, skel_image, display=False):
    """
    Starting with a morphological skeleton, creates a corresponding binary image
    with all end-points pixels (1) and all other pixels (0).
    """
    
    # identify 3-way branch-points through the hit-or-miss transform using
    # appropriate structure elements for an 8-connected skeleton:
    # http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm   
    hit = array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]], dtype=uint8)
    miss = array([[0, 0, 0],
                     [1, 0, 1],
                     [1, 1, 1]], dtype=uint8)

    # use some nifty NumPy slicing to add the three remaining rotations of each
    # of the structure elements to the hit and miss lists
    hit_list = [hit,]
    miss_list = [miss,]
    for ii in range(3):
        hit_list.append(transpose(hit_list[-1])[::-1, ...])
        miss_list.append(transpose(miss_list[-1])[::-1, ...])

    # create a zero array of the same shape as the skeleton and use it to collect
    # "hits" from the hit-and-miss operation
    end_points_image = zeros(bw_image.shape)
    for hit, miss in zip(hit_list, miss_list):
        curr = _hit_and_miss(skel_image, hit, miss)
        end_points_image = logical_or(end_points_image, curr)
    
    # pixels may "hit" multiple structure elements, ensure the output is a binary
    # image
    end_points_image = where(end_points_image, 1, 0)

    # use SciPy's ndimage module to label each contiguous foreground feature 
    # uniquely, this will locating and determining coordinates of each branch-point
    (labels, num_labels) = ndimage.label(end_points_image)
    
    # use SciPy's ndimage module to determine the coordinates/pixel corresponding
    # to the center of mass of each branchpoint
    end_points = ndimage.center_of_mass(bw_image, labels=labels, index=range(1, num_labels+1))
    end_points = array([value for value in end_points if not isnan(value[0]) or not isnan(value[1])], dtype=int)
    num_end_points = len(end_points)

    if display== True:
        pyplot.figure()
        pyplot.imshow(bw_image, cmap=pyplot.get_cmap("gray"))
        pyplot.scatter(end_points[:,1], end_points[:,0], marker='.', c="r", s=64, lw=0)
        pyplot.show()
        
    # return a list of end -oint coordinates (x, y) and the number of end-
    # points
    return(end_points, num_end_points)