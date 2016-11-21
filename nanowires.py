#!/usr/bin/env python
#
# Project: Image Processing for Quantification of Nanowire Alignment on Surfaces 
# File name: nanowires.py
# Description:  see "docstring" below
#   
# @author: N.M. Abukhdeir (University of Waterloo)
#         Copyright (C) 2016.
# @version: 0.2 
#   
# @see The GNU Public License (GPL)
#

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Sample script using the OpenCV library to quantify alignment of nanowires 
on a surface from experimental images.

This script uses the OpenCV library to compute the orientational order parameters 

S_2n = cos(2 n*\theta)

and implements the methods presented in

Dong, Goldthorpe, and Abukhdeir (2016) submitted to Nanotechnology

for a set of 1D nanowires on a surface.

Usage requires specification of the image directory (PNG format, *.png) and a 
directory for the output of the script:

      $ python nanowire.py images/ output/
"""

import os
import sys

from numpy import *
from scipy import ndimage
import cv2
from matplotlib import pyplot


################################################################################
# Utility Functions
def load_image(filename, display=False):
    "Load an individual image from a file and convert to grayscale."
    
    raw_image = cv2.imread(filename)

    if raw_image == None:
        print("load_image(): Image file not found: " + filename)
        raise IOError

    grey_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

    if display == True:
        cv2.imshow("grey_image", grey_image)
        cv2.waitKey()

    return(grey_image)


def digitize_image(grey_image, block_size, threshold="otsu", denoising_strength=30, display=False):
    "Digitize a loaded image by first applying a denoising filter and then using either Otsu or adaptive thresholding."
    
    # only apply the denoising filter if the denoising strength is greater than zero
    if denoising_strength > 0:
        filtered_image = cv2.fastNlMeansDenoising(grey_image, None, denoising_strength, 3*block_size[0], block_size[0])
    else:
        filtered_image = grey_image

    # perform the user-specified thresholding method to yield a 0/1 binary image
    if threshold == "otsu":
        (retval, binary_image) = cv2.threshold(filtered_image, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif threshold == "adaptive":
        # execute adaptive thresholding, results in a binary image
        binary_image = cv2.adaptiveThreshold(filtered_image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size[0], -1) 
    else:
        raise NotImplementedError

    if display == True:
        pyplot.figure()
        pyplot.imshow(binary_image, cmap=pyplot.get_cmap("gray"))
                
        pyplot.show()
            
    return(binary_image)


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


################################################################################
# 1D Nanostructure Functions

def find_segments(bw_image, skel_image, cutoff, prunings=0, display=False):
    """
    Given a binary image and its morphological skeleton, identify candidate
    1D nanostructure segments by subtracting branch-point pixels from the skeleton
    and then grouping end-points in the modified skeleton into pairs (segments).
    """
    
    # find branch points morphological skeleton
    (branch_points, num_branch_points) = find_branch_points(bw_image, skel_image, display=False)
    
    # create a deep copy of the morphological skeleton and subtract branch-point 
    # pixels and their nearest-neighbour pixels
    branchless_skel_image = skel_image.copy()
    for (x, y) in branch_points:
        indices = [-1, 0, 1]
        (upper1, upper2) = branchless_skel_image.shape
        for i in indices:
            for j in indices:
                x2 = x + i
                y2 = y + j
                if x2 >=0 and y2 >= 0 and x2 < upper1 and y2 < upper2:
                    branchless_skel_image[x2, y2] = 0
    
    # find end points in the branchless morphological skeleton
    (end_points, num_end_points) = find_end_points(bw_image, branchless_skel_image, display=False)
    
    # group end points into sets for each distinct feature, should be in sets of two
    (label_image, num_labels) = ndimage.label(branchless_skel_image, structure=ones((3,3),dtype=uint8))
    end_points_mapping = [[] for x in range(num_labels)]
    for i in range(num_end_points):
        label_index = label_image[end_points[i][0], end_points[i][1]] - 1
        end_points_mapping[label_index].append(end_points[i])   


    # filter features that have anything other than 2 end-points within them,
    # for those that have two end-points find the Euclidean distance between them
    # and filter if below cutoff    
    nw_segments = []
    filtered_segments = []
    for mapping in end_points_mapping:
        if len(mapping) == 2:
            (start, end) = mapping
            
            # take into account the number of prunings in the length
            length = sqrt((start[0]-end[0])**2 + (start[1]-end[1])**2) + 2*prunings
            
            if length > cutoff:
                nw_segments.append(mapping)
            else:
                filtered_segments.append(mapping) 
        else:
            filtered_segments.append(mapping)  

    if display != False:
        fig = pyplot.figure()
        pyplot.imshow(branchless_skel_image, cmap=pyplot.get_cmap("gray"))
        #pyplot.scatter(end_points[:,1], end_points[:,0], c="r")
        
        for (p1, p2) in nw_segments:
            pyplot.plot([p1[1], p2[1]], [p1[0], p2[0]],'b', linewidth=2)
        
        if display == True:
            pyplot.show()
        else:
            fig.savefig(display + "_segments.png")

    # return two lists of pairs of end-points, valid ones and filtered ones
    return(nw_segments, filtered_segments) 


def compute_order(nw_segments, order=6, display=False):
    """
    Given a set of nanostructure segments, compute orientational order parameters
    S_2n up to a specified order
    """
    
    # create lists to populate with approximate length and orientation angle (with
    # respect to average orientation vector)
    length_list = []
    orientation_list = []

    # iterate through the segments list and compute the polar angle of orientation,
    # because the segments are assumed non-polar, add both m and -m orientations
    for (start, end) in nw_segments:
        length = sqrt((start[0]-end[0])**2 + (start[1]-end[1])**2)
        
        mx = (start[0]-end[0])/length
        my = (start[1]-end[1])/length
        
        # don't mix this up, y before x!
        orientation1 = arctan2(my, mx)
        orientation2 = arctan2(-my, -mx)
                
        length_list.append(length)             
        orientation_list.append(orientation1 + pi)
        length_list.append(length)             
        orientation_list.append(orientation2 + pi)
    
    # create ndarray objects from the lists for numeric manipulations
    length = array(length_list)
    orientation = array(orientation_list) 
    orientation = where(orientation < 0, orientation + pi, orientation)
 
    # fit order parameters to orientation samples, use both unweighted and weighted order
    # parameters
    (S, Sw, orientation, orientation_weighted) = fit_odf_samples(orientation, length, order)
    
    # return ndarrays of unweighted (S) and weighted (Sw) order parameters, and
    # unweighted and weighted angles from average orientation vector
    return(S, Sw, length, orientation, orientation_weighted)


def fit_odf_samples(orientation, length, order, display=False):
    """
    Find orientational order parameters up to order from orientation angle samples
    for both unweighted and weighted order parameters.
    """
    # determine the alignment vectors for each nanowire, angles are with respect
    # to the horizontal axis of the image (polar coordinates)
    mx = cos(orientation)
    my = sin(orientation)
    
    # determine the average orientation vector, or "director", this uses a simple
    # method formulated in: R. Eppenga and D. Frenkel, Mol. Phys., 1984, 52, 1303-1334.
    Q = zeros((len(length), 2,2))
    Qw = zeros((len(length), 2,2))
    Q[:, 0, 0] = mx*mx - 0.5
    Q[:, 1, 1] = my*my - 0.5
    Q[:, 0, 1] = Q[:, 1, 0] = mx*my

    Qw[:, 0, 0] = (mx*mx - 0.5)*length
    Qw[:, 1, 1] = (my*my - 0.5)*length
    Qw[:, 0, 1] = Qw[:, 1, 0] = mx*my*length

    Q = Q.mean(axis=0)
    Qw = Qw.sum(axis=0)/length.sum()
    
    (evals, evecs) = linalg.eigh(Q)
    n = evecs[:,evals.argmax()]    

    (evals, evecs) = linalg.eigh(Qw)
    nw = evecs[:,evals.argmax()] 
       
    # determine orientation with respect to average orientation vector from above,
    # unweighted formulation
    orientation = arccos(n[0]*mx + n[1]*my)
    orientation = where(orientation > pi/2., orientation - pi, orientation)

    # determine orientation with respect to average orientation vector from above,
    # unweighted formulation
    orientation_weighted = arccos(nw[0]*mx + nw[1]*my)
    orientation_weighted = where(orientation_weighted > pi/2., orientation_weighted - pi, orientation_weighted)

    S = zeros((order/2,))
    Sw = zeros((order/2,))
    
    # compute the weighted and unweighted order parameters
    for i in range(len(S)):
        S[i] = (cos(2*(i+1)*orientation)).mean()
        Sw[i] = (cos(2*(i+1)*orientation_weighted)*length).sum()/length.sum()

    return(S, Sw, orientation, orientation_weighted)


def compute_odf(S, n=10000, display=False):
    """
    Given a set of orientational order parameters compute the corresponding 
    approximate orientational distribution function
    """
    
    # use a shifted polar coordinate system for convenient plotting
    theta = linspace(-pi/2., pi/2., n)
    
    p = zeros(theta.shape)
    
    # add the S_0 contribution
    p += 1./(2*pi)
    
    # add the S_2 and greater contributions
    for i in range(len(S)):
        p += S[i]*cos(2.*(i+1)*theta)/pi

    if display != False:
        fig = pyplot.figure()
        pyplot.plot(theta, p)

        if display == True:
            pyplot.show()
        else:
            fig.savefig(display + "_odf.png")

    # polar coordinate and p(\theta) value
    return(theta, p)
    
# functionality to execute if the module is being executed as a script
if __name__ == "__main__":
    
    # default image processing parameters parameters
    prunings = 0        # number of prunings applied to the morphological skeleton
    block_size = (7, 7) # block size for all filter operatings
    threshold = "otsu"  # thresholding method, either "otsu" or "adaptive"
    cutoff = 5          # nanostructure length cutoff in pixels

    order = 50          # number of order parameters to compute, S_{2n}

    Ntheta=100          # number of ODF values to compute on the interval [-\pi, \pi]
    nbins= 49           # number of bins for alignment histograms
    
    num_args = len(sys.argv)

    if num_args == 0:
        # use default image and data directories
        image_directory = "images/"
        data_directory = "data/"
    elif num_args == 3:
        image_directory = sys.argv[1]
        data_directory = sys.argv[2]
    else:
        print("Usage: %s image_directory/ data_directory/" % sys.argv[0])
        exit()

    print("Running alignment image processing script...")
    print("Image directory: %s" % image_directory)
    print("Data directory: %s" % data_directory)

    print("\nIterating over all PNG (*.png) files in the image directory...")
    
    for (root, dirs, files) in os.walk(image_directory):
        for filename in files:
            if(filename.endswith("png")):
                print("Processing file: ", root + filename)
               
                data_name = data_directory + filename.split('.')[0]              
                
                grey_image = load_image(root + filename, display=False)

                bw_image = digitize_image(grey_image, block_size, threshold=threshold, display=False)

                skel_image = skeletonize_image(bw_image, display=False)

                (nw_segments, filtered_segments) = find_segments(bw_image, skel_image, cutoff=cutoff, prunings=prunings, display=False)

                (S, Sw, length, orientation, orientation_weighted) = compute_order(nw_segments, order=order, display=False)

                (theta, p0) = compute_odf(S[:1], n=Ntheta)
                (theta, pw0) = compute_odf(Sw[:1], n=Ntheta)

                (theta, p) = compute_odf(S, n=Ntheta)
                (theta, pw) = compute_odf(Sw, n=Ntheta)

                Nsamples = len(orientation)
                weights = ones((Nsamples,))*(nbins/(2*pi))/(Nsamples)
                pyplot.figure()
                pyplot.hist(orientation, bins=nbins, range=(-pi/2, pi/2), weights=weights, histtype="bar")
                pyplot.plot(theta, p, 'r', linewidth=3)
                pyplot.plot(theta, p0, 'b--', linewidth=3)
                pyplot.ylim((-.1, 2.))
                pyplot.xlim((-pi/2., pi/2.))
                pyplot.ylabel(r"$p(\theta_i)$")
                pyplot.xlabel(r"$\theta_i$")
                pyplot.savefig(data_name + "_unweighted.pdf", bbox_inches="tight")

                Nsamples = len(orientation)
                weights = ones((Nsamples,))*(nbins/(2*pi))/(Nsamples)
                pyplot.figure()
                pyplot.hist(orientation_weighted, bins=nbins, range=(-pi/2, pi/2), weights=weights, histtype="bar")
                pyplot.plot(theta, pw, 'r', linewidth=3)
                pyplot.plot(theta, pw0, 'b--', linewidth=3)
                pyplot.ylim((-.1, 2.))
                pyplot.xlim((-pi/2., pi/2.))
                pyplot.ylabel(r"$p(\theta_i)$")
                pyplot.xlabel(r"$\theta_i$")
                pyplot.savefig(data_name + "_weighted.pdf", bbox_inches="tight")

                pyplot.figure()
                width= 0.5
                index = arange(2, order+2, 2)
                pyplot.plot(index + 2, S, 'r-o')
                pyplot.plot(index + 2, Sw, 'b-s')
                pyplot.xlim((2., order))
                pyplot.ylim((-.1, 1.))
                pyplot.ylabel(r"$S_n$")
                pyplot.xlabel(r"$n$")
                pyplot.savefig(data_name + "_order_params.pdf", bbox_inches="tight")

                pyplot.close("all")

