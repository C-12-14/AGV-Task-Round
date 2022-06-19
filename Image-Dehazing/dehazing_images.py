from dehazing import Dehazing
import cv2
import csv


def load_images():
    hazyimages = [
        cv2.imread(f"Image-Dehazing/Hazy/{i:02d}_outdoor_hazy.jpg")
        for i in range(1, 17)
    ]
    gtimages = [
        cv2.imread(f"Image-Dehazing/GT/{i:02d}_outdoor_GT.jpg") for i in range(1, 17)
    ]
    return hazyimages, gtimages


def dehaze_images():
    hazyimages, gtimages = load_images()
    dehazer = Dehazing()

    with open("Image-Dehazing/Dehazed/output.csv", "w") as f:
        writer = csv.writer(f, delimiter=",")

        for img1, img2, i in zip(hazyimages, gtimages, range(1, len(hazyimages) + 1)):
            imgdh = dehazer.dehazing(img1, 0.1, 200)
            p = dehazer.psnr(imgdh, img2)
            s = dehazer.ssim(imgdh, img2)
            writer.writerow([f"{i:02d}", p, s])
            cv2.imwrite(f"Image-Dehazing/Dehazed/{i:02d}_outdoor_dh.png", imgdh)
            print("awefihwiaefh")


dehaze_images()
