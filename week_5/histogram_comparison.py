import cv2
import numpy as np

def compare_histograms(histogram_1, histogram_2, method):
    methods = (
        cv2.HISTCMP_CORREL,
        cv2.HISTCMP_CHISQR,
        cv2.HISTCMP_INTERSECT,
        cv2.HISTCMP_BHATTACHARYYA,
    )
    method = methods[method]

    result = cv2.compareHist((histogram_1),(histogram_2), method)

    return result


def compare_descriptors(des1, kp2, des2):
    bf = cv2.BFMatcher()
    # Match descriptors.
    matches = bf.knnMatch(des1, trainDescriptors = des2, k = 2)
    similar = []
    if len(kp2)==0 or len(matches)==0:
        return 0
    try:
        for m,n in matches:
            if m.distance < 0.4 * n.distance:
                similar.append([m])

    except ValueError:
        return 0
    return (len(similar) / len(kp2))

