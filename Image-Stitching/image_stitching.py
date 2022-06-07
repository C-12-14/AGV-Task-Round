import numpy as np
import cv2
import matplotlib.pyplot as plt
from pprint import pprint


def sift(img1, img2):
    sift_detector = cv2.SIFT_create()
    k1, d1 = sift_detector.detectAndCompute(img1, None)
    k2, d2 = sift_detector.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.25 * n.distance:
            good.append([m])
    good = np.array(good)
    return k1, k2, good


def ransac(k1, k2, good, n=100):
    maxin = 0
    homography = 0

    for i in range(n):
        inliers, outliers = np.empty((0, 1)), np.empty((0, 1))
        ps = good[np.random.choice(good.shape[0], 2), :]
        A = []

        for p in ps:
            p = p[0]
            xs, ys = k1[p.queryIdx].pt
            xd, yd = k2[p.trainIdx].pt
            A.append([xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys, -xd])
            A.append([0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys, -yd])
        A = np.array(A)
        eigenvalue, eigenvector = np.linalg.eig((A.T) @ A)
        idx = eigenvalue.argsort()[::-1]
        eigenvector = eigenvector[:, idx]
        h = np.reshape(eigenvector[:, -1], (3, 3))

        for point in good:
            point = point[0]
            c1 = np.array(k1[point.queryIdx].pt)
            c1 = np.append(c1, [1])
            c2 = np.array(k2[point.trainIdx].pt)
            c2 = np.append(c2, [1])

            pred_c2 = h @ c1.T
            pred_c2 /= pred_c2[2]
            diff = np.linalg.norm(pred_c2 - c2)

            if diff < 100:
                inliers = np.append(inliers, np.array([point]))
            else:
                outliers = np.append(outliers, np.array([point]))

        if maxin < inliers.shape[0]:
            maxin = inliers.shape[0]
            homography = h

    return homography, inliers, outliers


# def stitching(img1, img2, )
def main():
    img1 = cv2.imread("Image-Stitching/images/test1.png")
    img2 = cv2.imread("Image-Stitching/images/test2.png")
    k1, k2, good = sift(img1, img2)
    imgout1 = cv2.drawMatchesKnn(
        img1,
        k1,
        img2,
        k2,
        good,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(imgout1)
    homography, inliers, outliers = ransac(k1, k2, good, 1000)
    inliers = np.reshape(inliers, (inliers.shape[0], 1))
    outliers = np.reshape(outliers, (outliers.shape[0], 1))
    imgout2 = cv2.drawMatchesKnn(
        img1,
        k1,
        img2,
        k2,
        inliers,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
    )
    print(inliers.shape[0])
    print(good.shape[0])
    ax[1].imshow(imgout2)
    plt.show()


main()
