import numpy as np
import cv2
import sys
import cython_file

def getImage():
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        image = cv2.imread(filename)
    elif len(sys.argv) < 2:
        cap = cv2.VideoCapture(0)
        for i in range(0,15):
            retval,image = cap.read()
        if retval:
            cv2.imwrite("capture.jpg", image)
    print(image.shape)
    return image

def reload(img):
    return img

def save(img):
    cv2.imwrite("new.jpg",img)

def cvt2Gry1(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def cvt2Gry2(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray/255

def cvt2Gry3(img):
    img = img.astype(np.int32)
    img = cython_file.compute(img)
    img = np.array(img)/255
    return img

def cycleColor(image, i):
    img1=image.copy()
    if i%3 == 0:
        img1[:,:,1] = 0
        img1[:,:,2] = 0
        show(img1)
    elif i%3 == 1:
        img1[:,:,0] = 0
        img1[:,:,2] = 0
        show(img1)
    elif i%3 == 2:
        img1[:,:,1] = 0
        img1[:,:,0] = 0
        show(img1)


def rotation(img):
    def sliderHandler(degree):
        cols = image.shape[0]
        rows = image.shape[1]
        M = cv2.getRotationMatrix2D((rows / 2, cols / 2), degree, 1)
        dst = cv2.warpAffine(image, M, (rows, cols))
        cv2.imshow("Display window", dst)
    image = cvt2Gry1(img)
    cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Angle", "Display window", 0, 360, sliderHandler)
    cv2.imshow("Display window", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show(img):
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def program_desc():
    print("""
    press i to reload the original image
    press w to save the current image
    press g to convert the image to grayscale
    press G to convert the image to grayscale using conversion function
    press T to convert the image to grayscale using cython
    press c to cycle through color channels
    press r to roate the image
    press h for program_desc
    press s for viewing image""")

def main():
    image = getImage()
    img = image.copy()
    i=0
    k = input()
    print("input key to Process image(press 'q' to quit):")
    while k != 'q':
        if k == 'i':
            image = reload(img)
        elif k == 'w':
            save(image)
        elif k == 'g':
            image = cvt2Gry1(image)
        elif k == 'G':
            image = cvt2Gry2(image)
        elif k == 'T':
            image = cvt2Gry3(image)
        elif k == 'c':
            cycleColor(image, i)
            i +=1
        elif k == 'r':
            image = rotation(image)
        elif k == 'h':
            program_desc()
        elif k == 's':
            show(image)
        k = input()

if __name__ =="__main__":
    main()