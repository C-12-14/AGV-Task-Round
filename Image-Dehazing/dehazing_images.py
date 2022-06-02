import numpy as np
import cv2
import glob
from pprint import pprint
from scipy.ndimage import gaussian_filter
import csv


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    return 20 * np.log10(255) - 10 * np.log10(mse)


def ssim(im1, im2):
    k1 = 0.01
    k2 = 0.03
    channels = im1.shape[2]
    mssim = np.empty(channels, dtype=np.float64)
    i1 = cv2.split(im1)
    i2 = cv2.split(im2)
    for img1, img2, i in zip(i1, i2, range(channels)):
        img1 = img1.astype(np.float64, copy=False)
        img2 = img2.astype(np.float64, copy=False)
        # Gaussian weighted average
        ux = gaussian_filter(img1, 1.5, mode="reflect", truncate=3.5)
        uy = gaussian_filter(img2, 1.5, mode="reflect", truncate=3.5)
        uxx = gaussian_filter(img1 * img1, 1.5, mode="reflect", truncate=3.5)
        uyy = gaussian_filter(img2 * img2, 1.5, mode="reflect", truncate=3.5)
        uxy = gaussian_filter(img1 * img2, 1.5, mode="reflect", truncate=3.5)
        # Variance
        sx = uxx - ux * ux
        sy = uyy - uy * uy
        sxy = uxy - ux * uy

        L = 255
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2

        A1, A2, B1, B2 = (
            2 * ux * uy + c1,
            2 * sxy + c2,
            ux**2 + uy**2 + c1,
            sx + sy + c2,
        )
        D = B1 * B2
        result = (A1 * A2) / D
        mssim[i] = result.mean()
    return mssim.mean()


def dehazing(img, delta=0.9, sigma=1):
    minIc = img.min(axis=2).astype(np.float64, copy=False)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float64, copy=False)
    gray = gray.ravel()
    t10 = int(len(gray) * 0.01)
    A = gray[np.argpartition(gray, -t10)[-t10:]].mean()
    MAX = minIc.max()
    MIN = minIc.min()
    J = delta * (minIc - MIN) * minIc / (MAX - MIN)
    t = abs((A - minIc) / (A - J))
    maxt = t.max()
    t[minIc > A] = t[minIc > A] / maxt
    t = gaussian_filter(t, sigma)
    t[t < 0.05] = 0.05
    t = np.repeat(t[:, :, None], 3, axis=2)
    J = ((img - A) / t) + A + 15
    J[J > 255] = 255
    J[J < 0] = 0
    J = J.astype(np.uint8, copy=False)
    return J


# im = cv2.imread("Image-Dehazing/test5.png")
# imd = dehazing(im, 0.95, sigma=1)
# cv2.imwrite("Dehazed01.png", im)
# cv2.imshow("im dehazed", imd)
# cv2.imshow("im", im)
# im = cv2.imread("Image-Dehazing/Hazy/08_outdoor_hazy.jpg", 33)
# imd = dehazing(im, 0.95, sigma=1)
# # cv2.imwrite("Dehazed01.png", im)
# cv2.imshow("im2 dehazed", imd)
# cv2.imshow("im2", im)
# imd = dehazing(im, 0.8, sigma=1)
# cv2.imshow("im2 dehazed2", imd)
# while True:
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break


def main():
    hazyimages = [
        cv2.imread(f"Image-Dehazing/Hazy/{i:02d}_outdoor_hazy.jpg")
        for i in range(1, 17)
    ]
    gtimages = [
        cv2.imread(f"Image-Dehazing/GT/{i:02d}_outdoor_GT.jpg") for i in range(1, 17)
    ]
    with open("Image-Dehazing/output.csv", "w") as f:
        writer = csv.writer(f, delimiter=",")
        for img1, img2, i in zip(hazyimages, gtimages, range(1, len(hazyimages) + 1)):
            imgdh = dehazing(img1, 1, 7)
            p = psnr(imgdh, img2)
            s = ssim(imgdh, img2)
            print(s)
            writer.writerow([f"{i:02d}", p, s])
            cv2.imwrite(f"Image-Dehazing/Dehazed/{i:02d}_outdoor_dh.png", imgdh)
