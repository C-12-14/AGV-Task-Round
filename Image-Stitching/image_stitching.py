import numpy as np
import cv2
import matplotlib.pyplot as plt
from pprint import pprint
import scipy.ndimage
import os

dirname = os.path.dirname(__file__)


class ImageStitching:
    def init(self):
        pass

    def sift(self, img1: np.ndarray, img2: np.ndarray, thres: float = 0.5):
        """Using SIFT to detect and match features between images.

        :param np.ndarray img1: first image
        :param np.ndarray img2: second image
        :param float thres: threshold for comparing the pair of matches corresponding to each keypoint given by bf.knnMatch()

        :returns:
            - k1: keypoints corresponding to img1
            - k2: keypoints corresponding to img2
            - good: matches above a certain threshold
        """

        sift_detector = cv2.SIFT_create()
        k1, d1 = sift_detector.detectAndCompute(img1, None)
        k2, d2 = sift_detector.detectAndCompute(img2, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(d1, d2, k=2)

        good = []
        for m, n in matches:
            if m.distance <= thres * n.distance:
                good.append([m])
        good = np.array(good)
        return k1, k2, good

    def ransac(self, k1, k2, good, n: int = 100):
        """Using ransac to find the best homography between two images.

        :param k1: keypoints corresponding to first image
        :param k2: keypoints corresponding to second image
        :param good: matches between k1 and k2
        :param n: number of iterations

        :returns:
            - np.ndarray homography: homography corresponding to the most number of inliers
            - np.ndarray inliers: matches that are inliers according to the best homography
            - np.ndarray outliers: matches ethat are outliers according to the best homography

        """
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

    def image_stitching(self, img1: np.ndarray, img2: np.ndarray, homography):
        """Stitching the two images using the provided homography

        :param np.ndarray img1: first image to be stitched
        :param np.ndarray img2: second image to be stitched
        :param np.ndarray homography: homography to be used

        :return: stitched image
        :rtype: np.ndarray
        """

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

    def plotting(
        self, img1, img2, k1, k2, inliers, outliers, stitched_image, name="Input Image"
    ):
        """Plotting the matches and final stitched image for visualisation

        :param np.ndarray img1 : image to be stitched
        :param np.ndarray img2 : image to be stitched
        :param k1 : keypoints detected by SIFT in img1
        :param k2 : keypoints detected by SIFT in img2
        :param np.ndarray inliers : inlier matches corresponding to the best homography
        :param np.ndarray outliers : outlier matches corresponding to the best homography
        :param np.ndarray stitched_image : final stitched image
        :param str name: Name to be displayed in the Title
        """

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
        # fig.savefig(os.path.join(dirname, f"output/{name}_plot.png"), dpi=300)
        plt.show()


if __name__ == "__main__":
    stitcher = ImageStitching()
    name = "foto1"
    img1 = cv2.imread(os.path.join(dirname, f"images/{name}B.jpg"))
    img2 = cv2.imread(os.path.join(dirname, f"images/{name}A.jpg"))

    k1, k2, good = stitcher.sift(img1, img2, thres=0.3)
    homography, inliers, outliers = stitcher.ransac(k1, k2, good, 100)
    stitched_image = stitcher.image_stitching(img1, img2, homography)
    # cv2.imwrite(os.path.join(f"output/{name}.jpg", stitched_image))
    stitcher.plotting(img1, img2, k1, k2, inliers, outliers, stitched_image, name=name)
