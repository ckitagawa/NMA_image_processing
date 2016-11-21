"""
Main script
"""

import transform_lib as tl
import preprocessing as pp
import cv2

if __name__ == "__main__":

    # Process image.
    img = cv2.imread('corneal1.png', 0)
    img = pp.preprocess(img)
    
    # Detect blobs.
    detector = tl.BlobDetector()
    detector.set_defaults()
    keypoints = detector.detect_keypoints(img)
    pts = [x.pt for x in keypoints]

    # Filter points to centeral area.
    filtered_pts = list(filter(lambda x: x[0] < 450 and x[0] > 255, pts))

    # Analyze response.
    analyzer = tl.DelaunayAnalyzer(filtered_pts)
    analyzer.boo_graph()
    #analyzer.show(range(1, 11))



