from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import measure
from skimage import filters
import cv2


def main():
    fname='TNBC_NucleiSegmentation/GT_01/01_1.png'
    blur_radius = 1.0
    threshold = 50

    img = Image.open(fname).convert('L')
    img = np.asarray(img)
    print(img.shape)
    # (160, 240)

    # smooth the image (to remove small objects)
    imgf = ndimage.gaussian_filter(img, blur_radius)
    threshold = 50

    # find connected components
    labeled, nr_objects = ndimage.label(imgf > threshold)
    print("Number of objects is {}".format(nr_objects))
    # Number of objects is 4



    im = ndimage.gaussian_filter(img, blur_radius)
    all_labels, nr_objects2 = measure.label(im >  threshold, connectivity=1, return_num=True)

    # Threshold it so it becomes binary
    ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # You need to choose 4 or 8 for connectivity type
    connectivity = 4
    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    plt.figure(figsize=(9, 3.5))
    plt.subplot(141)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.subplot(142)
    plt.title('Scipy-Ndimage {}'.format(nr_objects))
    plt.imshow(labeled, cmap='nipy_spectral')
    plt.axis('off')
    plt.subplot(143)
    plt.title('Skimage-Measure {}'.format(nr_objects2))
    plt.imshow(all_labels, cmap='nipy_spectral')
    plt.axis('off')
    plt.subplot(144)
    plt.title('OpenCV {}'.format(num_labels))
    plt.imshow(labels, cmap='nipy_spectral')
    plt.axis('off')


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
