import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

dirname = os.path.dirname(__file__)


class line:
    def __init__(self, m, c, inliers, outliers):
        """
        :param float m: slope of the line
        :param float c: y intercept of the line
        :param np.ndarrays inliers: array of all the inlier blobs
        :param np.ndarrays outliers: array of all the outlier blobs
        """
        self.m = m
        self.c = c
        self.inliers = inliers
        self.outliers = outliers


def point_detection(img):
    """Blob detection in the image
    :param np.ndarray img

    :return: coordinates
    :rtype: np.ndarray
    """
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 20
    params.maxThreshold = 200
    params.filterByCircularity = False
    params.filterByConvexity = False

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)

    coordinates = np.array([[point.pt[0], point.pt[1]] for point in keypoints])
    return coordinates


def plot(linefit, img):
    """Plotting all the points with the line and its inliers and outliers

    :param line
    """
    w, h = img.shape
    fig, ax = plt.subplots(figsize=(h / 100, w / 100))
    f = lambda x: linefit.m * x + linefit.c
    x = np.linspace(0, w, 2)
    y = f(x)
    plt.style.use(["fivethirtyeight"])
    ax.plot(
        x,
        y,
        color="g",
        alpha=0.3,
        lw=3,
        label=f"y ={round(linefit.m, 4)}x+{round(linefit.c)}",
    )
    ax.plot(
        linefit.inliers[:, 0],
        linefit.inliers[:, 1],
        "o",
        color="g",
        label=f"Inliers: {len(linefit.inliers)}",
    )
    ax.plot(
        linefit.outliers[:, 0],
        linefit.outliers[:, 1],
        "o",
        color="r",
        label=f"Outliers: {len(linefit.outliers)}",
    )
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.legend(fontsize=10)
    ax.xaxis.tick_top()
    ax.set_title("2D Line Fitting Task", fontsize=18)
    plt.gca().invert_yaxis()
    fig.tight_layout()
    # plt.savefig(os.path.join(dirname, "output_after_100_iterations.png"), dpi=300)
    plt.show()


def ransac_line(coordinates, n=20, r=8):
    """
    :param int n: number of iterations
    :param float r: maximum distance around the line where inliers can be present

    :return: bestfit
    :rtype: line
    """
    results = np.array([])
    maxin = 0
    bestfit = 0
    for _ in range(n):
        inliers, outliers = np.empty((0, 2)), np.empty((0, 2))
        p1, p2 = coordinates[np.random.randint(coordinates.shape[0], size=2), :]
        if (p2[0] - p1[0]) == 0:
            continue
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        c = p1[1] - m * p1[0]
        y = lambda x: m * x + c
        dist = lambda px, py: abs((y(px) - py) / np.sqrt(1 + m**2))

        for point in coordinates:
            if dist(point[0], point[1]) <= r:
                inliers = np.append(inliers, np.array([[point[0], point[1]]]), axis=0)
            else:
                outliers = np.append(outliers, np.array([[point[0], point[1]]]), axis=0)
        if maxin < len(inliers):
            maxin = len(inliers)
            bestfit = line(m, c, inliers, outliers)
    return bestfit


if __name__ == "__main__":
    dots = []
    img = cv2.imread(os.path.join(dirname, "line_ransac.png"), cv2.IMREAD_GRAYSCALE)
    coor = point_detection(img)
    best = ransac_line(coor, 100, 20)
    plot(best, img)
