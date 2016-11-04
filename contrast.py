import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

params = cv2.SimpleBlobDetector_Params()

kernel = np.ones((2, 2), np.uint8)

img = cv2.imread('corneal1.png', 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)

dst = cv2.fastNlMeansDenoising(cl1)

#dil = cv2.dilate(dst,kernel, iterations=2)

blur = cv2.bilateralFilter(dst, 5, 75, 75)

thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 3)

erosion = cv2.erode(thresh, kernel, iterations=0)

#lines = cv2.HoughLines(dil, 2, 45, 10)

#median = cv2.medianBlur(dst,9)

params.minThreshold = 0.5
params.maxThreshold = 100

# Filter by Area.
params.filterByArea = True
params.minArea = 0.5

# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.4

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(erosion)
print(len(keypoints))

pts = [x.pt for x in keypoints]
filtered_pts = list(filter(lambda x: x[0] < 450 and x[0] > 225, pts))
tri = Delaunay(filtered_pts)
x = [pt[0] for pt in filtered_pts]
y = [pt[1] for pt in filtered_pts]

# plt.triplot(x, y, tri.simplices.copy())
# plt.plot(x, y, 'o')
# plt.show()


def find_neighbours(pindex, triang):
    return triang.vertex_neighbor_vertices[1][triang.vertex_neighbor_vertices[
        0][pindex]:triang.vertex_neighbor_vertices[0][pindex + 1]]


def boo_psi(points, n, triang):
    psi = []
    for pindex, point in enumerate(points):
        neighbours = find_neighbours(pindex, triang)
        N = len(neighbours)
        val = 0
        for _, index in enumerate(neighbours):
            a = [points[index][0] - point[0], points[index][1] - point[1]]
            val += np.exp(n * 1j * np.arccos(
                np.dot(a, [0, 1]) /
                (np.linalg.norm(a) * np.linalg.norm([0, 1]))))
        psi.append(np.linalg.norm(val / N))
    return psi


dx = range(1, 11)
dy = []
for x in dx:
    dy.append(np.average(boo_psi(filtered_pts, x, tri)))

plt.plot(dx, dy, 'o')
plt.show()

imk = cv2.drawKeypoints(erosion, keypoints, np.array([]), (0, 255, 0))
cv2.imwrite('res.jpg', imk)


##############################################################################
def draw_voronoi(img, subdiv):

    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))

        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Red is a nice colour
points_color = (0, 0, 255)

# Make a copy
img = imk.copy()

# Rectangle to be used with Subdiv2D
size = img.shape
rect = (0, 0, size[1], size[0])

# Create an instance of Subdiv2D
subdiv = cv2.Subdiv2D(rect)

# Create an array of points.
points = []

# in case the microscope program wants to store the points in a separate file
# ^ could save time by not doing the inital processing
"""
# Read in the points from a text file
with open("points.txt") as file :
    for line in file :
        x, y = line.split()
        points.append((int(x), int(y)))
"""
# list comprehension this shit
for i in range(len(keypoints)):
    points.append((keypoints[i].pt[0], keypoints[i].pt[1]))
# Insert points into subdiv
for p in points:
    subdiv.insert(p)

# Create space for Voronoi Diagram
img_voronoi = np.zeros(img.shape, dtype=img.dtype)

# Draw Voronoi diagram
draw_voronoi(img_voronoi, subdiv)

# Show results
cv2.imwrite('jesus_wants_me.jpg', img_voronoi)
