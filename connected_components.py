from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import measure
from skimage import filters
import cv2


def main():
    fname1='TNBC_NucleiSegmentation/GT_01/01_1.png'

    img1 = Image.open(fname1).convert('L')
    img1 = np.asarray(img1)

    # Threshold it so it becomes binary
    ret1, thresh1 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # You need to choose 4 or 8 for connectivity type
    connectivity1 = 8
    # Perform the operation
    output1 = cv2.connectedComponentsWithStats(thresh1, connectivity1, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels1 = output1[0]
    # The second cell is the label matrix
    labels1 = output1[1]
    # The third cell is the stat matrix
    stats1 = output1[2]
    # The fourth cell is the centroid matrix
    centroids1 = output1[3]



    fname2='TNBC_NucleiSegmentation/GT_01/01_1.png'

    img2 = Image.open(fname2).convert('L')
    img2 = np.asarray(img2)

    # Threshold it so it becomes binary
    ret2, thresh2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # You need to choose 4 or 8 for connectivity type
    connectivity2 = 8
    # Perform the operation
    output2 = cv2.connectedComponentsWithStats(thresh2, connectivity2, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels2 = output2[0]
    # The second cell is the label matrix
    labels2 = output2[1]
    # The third cell is the stat matrix
    stats2 = output2[2]
    # The fourth cell is the centroid matrix
    centroids2 = output2[3]


    plt.figure(figsize=(9, 3.5))
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.subplot(132)
    plt.title('OpenCV {}'.format(num_labels1))
    plt.imshow(labels1, cmap='nipy_spectral')
    plt.axis('off')
    plt.subplot(133)
    plt.title('OpenCV {}'.format(num_labels2))
    plt.imshow(labels2, cmap='nipy_spectral')
    plt.axis('off')



    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
