
import cv2
import sys
from numpy import dtype
from sewar.full_ref import ssim, psnr


if len(sys.argv) == 3:
    img_path = sys.argv[1]
    img_pred_path = sys.argv[2]
elif len(sys.argv) < 4:
    img_path = '00000_img.png'
    img_pred_path = "out.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img,(256,256))
img_pred = cv2.imread(img_pred_path)


print("SSIM: ",ssim(img,img_pred))
print("PSNR: ",psnr(img,img_pred))
