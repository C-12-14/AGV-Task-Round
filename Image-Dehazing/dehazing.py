import numpy as np
import cv2
import glob
from scipy.ndimage import gaussian_filter


class Dehazing:
    def __init__(
        self,
        delta: float = 0.9,
        sigma: float = 1,
        brightness_gain: float = 1.275,
        kernel: tuple = (301, 301),
    ):
        """
        :param float delta: Control factor
        :param float sigma: Variance for the gaussian distribution
        :param float brightness_gain: Gain in brighness after recovering the dehazed image
        :param tuple kernel: Size of kernel to be used when applying the gaussian filter
        """
        self.delta = delta
        self.sigma = sigma
        self.bgain = brightness_gain
        self.kernel = kernel

    def psnr(self, img1: np.ndarray, img2: np.ndarray):
        """Peak Signal to Noise Ratio

        :param np.ndarray img1: ndarray of first image
        :param np.ndarray img2: ndarray of second image

        :return: pnsr value
        :rtype: float
        """
        mse = ((img1 - img2) ** 2).mean()
        return 20 * np.log10(255) - 10 * np.log10(mse)

    def ssim(self, im1: np.ndarray, im2: np.ndarray):
        """Calculating 'Structural Similarity' between two images (im1, im2)

        :param np.ndarray im1: ndarray of first image
        :param np.ndarray im2: ndarray of second image

        :return: structural similarity
        :rtype: float
        """
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

            L = 2**8 - 1
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

    def atmospheric_light(self, img: np.ndarray, x: float = 0.05):
        """To find atmospheric lighting in the image.

        :param np.ndarray img: hazy image
        :param float x: for finding top x% brightest pixels

        :return: A
        :rtype: float
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Grayscale transformation
        gray = gray.ravel()
        top = int(len(gray) * x / 100)  # Selecting top x% bright pixels

        """Gray Value Calculation"""
        A = gray[
            np.argpartition(gray, -top)[-top:]
        ].mean()  # Mean of top x% bright pixels
        return A

    def transmission_map(self, A: float, minIc: np.ndarray):
        """To find the transmission map of the image

        :param float A: Atmospheric Light
        :param np.ndarray minIc: minimum color channel

        :return: t
        :rtype: np.ndarray
        """
        MAX = minIc.max()
        MIN = minIc.min()
        J = self.delta * (minIc - MIN) * minIc / (MAX - MIN)
        t = abs((A - minIc) / (A - J))
        maxt = t.max()
        t[minIc > A] = t[minIc > A] / maxt
        t = cv2.GaussianBlur(t, self.kernel, self.sigma, sigmaY=self.sigma)
        t[t < 0.05] = 0.05
        t = np.repeat(t[:, :, None], 3, axis=2)
        return t

    def recover_image(self, img: np.ndarray, A: float, t: np.ndarray):
        """To recover the final image given the Atmospheric lighting and the transmission map

        :param np.ndarray img: Original hazy image
        :param float A: Atmospheric Light
        :param np.ndarray t: Transmission map

        :return: J
        :rtype: np.ndarray
        """
        J = ((img - A) / t) + A * self.bgain
        J[J > 255] = 255
        J[J < 0] = 0
        J = J.astype(np.uint8, copy=False)
        return J

    def dehaze(
        self,
        img: np.ndarray,
    ):
        """Implementation of dehazing algorithm given in 'Fast Image Dehazing Method Based on Linear Transformation'.
           DOI:http://dx.doi.org/10.1109/TMM.2017.2652069

        :param ndarray img: ndarray of input image

        :return: Dehazed image (J)
        :rtype: np.ndarray
        """

        minIc = img.min(axis=2).astype(np.float64, copy=False)  # Minimum filtering
        A = self.atmospheric_light(img, x=0.05)
        t = self.transmission_map(A, minIc)
        J = self.recover_image(img, A, t)
        return J


# if __name__ == "__main__":
