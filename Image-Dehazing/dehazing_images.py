from dehazing import Dehazing
import cv2
import csv
import time
import os

dirname = os.path.dirname(__file__)


def load_images():
    hazyimages = [
        cv2.imread(os.path.join(dirname, f"Hazy/{i:02d}_outdoor_hazy.jpg"))
        for i in range(1, 17)
    ]
    gtimages = [
        cv2.imread(os.path.join(dirname, f"GT/{i:02d}_outdoor_GT.jpg"))
        for i in range(1, 17)
    ]
    return hazyimages, gtimages


def dehaze_images():
    hazyimages, gtimages = load_images()
    dehazer = Dehazing(delta=0.4, sigma=100, brightness_gain=1.225, kernel=(501, 501))

    with open(os.path.join(dirname, "Dehazed/output.csv"), "w") as f:
        writer = csv.writer(f, delimiter=",")

        for img1, img2, i in zip(hazyimages, gtimages, range(1, len(hazyimages) + 1)):
            f = time.time()
            imgdh = dehazer.dehaze(img1)
            e = time.time()
            p = dehazer.psnr(imgdh, img2)
            s = dehazer.ssim(imgdh, img2)
            print(s)
            writer.writerow([f"{i:02d}", p, s])
            cv2.imwrite(os.path.join(dirname, f"Dehazed/{i:02d}_outdoor_dh.png"), imgdh)


if __name__ == "__main__":
    dehaze_images()
