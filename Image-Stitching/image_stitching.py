import numpy as np
import cv2
import matplotlib.pyplot as plt
from pprint import pprint
import scipy.ndimage


def sift(img1, img2):
    sift_detector = cv2.SIFT_create()
    k1, d1 = sift_detector.detectAndCompute(img1, None)
    k2, d2 = sift_detector.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])
    good = np.array(good)
    return k1, k2, good


def ransac(k1, k2, good, n=100):
    maxin = 0
    homography = 0

    for i in range(n):
        inliers, outliers = np.empty((0, 1)), np.empty((0, 1))
        ps = good[np.random.choice(good.shape[0], 4), :]
        A = []

        for p in ps:
            p = p[0]
            xs, ys = k1[p.queryIdx].pt
            xd, yd = k2[p.trainIdx].pt
            A.append([-xs, -ys, -1, 0, 0, 0, xd * xs, xd * ys, xd])
            A.append([0, 0, 0, -xs, -ys, -1, yd * xs, yd * ys, yd])
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
            h_test = np.linalg.inv(h)

            pred_c1 = h_test @ c2.T
            pred_c1 /= pred_c1[2]
            pred_c1 = np.round(pred_c1)
            diff = np.linalg.norm(pred_c1 - c1)

            if diff <= 1:
                inliers = np.append(inliers, np.array([point]))
            else:
                outliers = np.append(outliers, np.array([point]))

        if maxin < inliers.shape[0]:
            maxin = inliers.shape[0]
            homography = h
    homography = np.linalg.inv(homography)
    return homography, inliers, outliers


def image_stitching(img1, img2, homography):
    topright = homography @ np.array([img2.shape[1], 0, 1]).T
    topright = topright / topright[2]
    bottomright = homography @ np.array([img2.shape[1], img2.shape[0], 1]).T
    bottomright = bottomright / bottomright[2]
    w = max(int(topright[0] / topright[2]), int(bottomright[0] / bottomright[2]))
    w = max(w, img1.shape[1])
    h = max(img1.shape[0], img2.shape[0])
    stitched = cv2.warpPerspective(img2, homography, (w, h))
    stitched[0 : img1.shape[0], 0 : img1.shape[1]] = img1
    return stitched


def plotting(img1, img2, k1, k2, inliers, outliers, stitched_image, name):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    sift_comparision = cv2.drawMatches(
        img1,
        k1,
        img2,
        k2,
        inliers,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 35, 35),
    )
    fig.suptitle(f"{name}", fontsize=20)
    ax[0].imshow(sift_comparision)
    ax[0].set_title("Matches of Best Homography")
    ax[1].imshow(stitched_image)
    ax[1].set_title("Stitched Image")
    fig.tight_layout()
    fig.savefig(f"Image-Stitching/output/{name}_plot.png", dpi=300)
    plt.show()


def main():
    name = "foto5"
    img1 = cv2.imread(f"Image-Stitching/images/{name}B.jpg")
    img2 = cv2.imread(f"Image-Stitching/images/{name}A.jpg")

    k1, k2, good = sift(img1, img2)
    homography, inliers, outliers = ransac(k1, k2, good, 100)
    stitched_image = image_stitching(img1, img2, homography)
    cv2.imwrite(f"Image-Stitching/output/{name}.jpg", stitched_image)
    plotting(img1, img2, k1, k2, inliers, outliers, stitched_image, name=name)


main()
