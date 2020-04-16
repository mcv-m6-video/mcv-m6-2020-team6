import cv2
import numpy as np


def parse_3_3_matrix_line(line, first_line_jump=0):
    splitLine = line.split(";")
    f_row = splitLine[0]
    s_row = splitLine[1]
    t_row = splitLine[2]

    f_row = f_row.split(" ")
    first_row = np.array(
        [
            float(f_row[first_line_jump]),
            float(f_row[first_line_jump + 1]),
            float(f_row[first_line_jump + 2]),
        ]
    )

    s_row = s_row.split(" ")
    second_row = np.array([float(s_row[0]), float(s_row[1]), float(s_row[2])])

    t_row = t_row.split(" ")
    third_row = np.array([float(t_row[0]), float(t_row[1]), float(t_row[2])])

    H = np.stack((first_row, second_row, third_row))

    return H


def parse_coefficients(line, num_coefficient=1, jump=0):
    values = line.split(" ")
    if num_coefficient == 1:
        return float(values[jump])
    else:
        float_values = [float(v) for v in values[jump:]]
        return np.array(float_values)


def read_homography(directory_txt):
    txt_gt = open(directory_txt, "r")

    lines = txt_gt.readlines()
    dict_return = {}

    for line in lines:
        if line.startswith("Homography"):
            dict_return["homography"] = parse_3_3_matrix_line(line, first_line_jump=2)

        if line.startswith("Intrinsic"):
            dict_return["instrinsic_parameters"] = parse_3_3_matrix_line(
                line, first_line_jump=3
            )

        if line.startswith("Distortion"):
            dict_return["distortion"] = parse_coefficients(
                line, num_coefficient=4, jump=2
            )

        if line.startswith("Reprojection"):
            dict_return["reprojection_error"] = parse_coefficients(
                line, num_coefficient=1, jump=2
            )

    return dict_return


if __name__ == "__main__":
    directory_txt = "dataset/c004/calibration.txt"
    h, d = read_homography(directory_txt)

    print("homo:", h)
    print("dist:", d)

"""
image = cv2.imread("dataset/c005/frames/image00000.jpg")

directory_txt = "dataset/c005/calibration.txt"

txt_gt = open(directory_txt, "r")
for line in txt_gt:
    splitLine = line.split(";")
    f_row = splitLine[0]
    s_row = splitLine[1]
    t_row = splitLine[2]

    f_row = f_row.split(" ")
    first_row = np.array([float(f_row[0]), float(f_row[1]), float(f_row[2])])

    s_row = s_row.split(" ")
    second_row = np.array([float(s_row[0]), float(s_row[1]), float(s_row[2])])

    t_row = t_row.split(" ")
    third_row = np.array([float(t_row[0]), float(t_row[1]), float(t_row[2])])

    h = np.stack((first_row, second_row, third_row))
    im_out = cv2.warpPerspective(image, h, (2000, 2000))

    # Display images
    cv2.imshow("Source Image", image)

    cv2.imshow("Warped Source Image", im_out)
    cv2.waitKey()
"""
