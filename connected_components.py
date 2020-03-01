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
    ret1, thresh1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)  # ensure binary
    # You need to choose 4 or 8 for connectivity type
    connectivity1 = 8
    # Perform the operation
    output1 = cv2.connectedComponentsWithStats(thresh1, connectivity1, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels1 = output1[0]
    # The second cell is the label matrix
    original_labels1 = output1[1]
    # The third cell is the stat matrix
    stats1 = output1[2]
    # The fourth cell is the centroid matrix
    centroids1 = output1[3]



    fname2='TNBC_NucleiSegmentation/Slide_01/01_1_result.png'

    img2 = Image.open(fname2).convert('L')
    img2 = np.asarray(img2)

    # Threshold it so it becomes binary
    ret2, thresh2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)  # ensure binary
    # You need to choose 4 or 8 for connectivity type
    connectivity2 = 8
    # Perform the operation
    output2 = cv2.connectedComponentsWithStats(thresh2, connectivity2, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels2 = output2[0]
    # The second cell is the label matrix
    original_labels2 = output2[1]
    # The third cell is the stat matrix
    stats2 = output2[2]
    # The fourth cell is the centroid matrix
    centroids2 = output2[3]


    #True Positive = cell in real / cell in the result
    #False Positive = not cell in real / cell in the result
    #False Negative = cell in real / not cell in the result
    #True Negative = not cell in real / not cell in the result
    #Total Elements = number of components in the GT


    labels1 = np.where(original_labels1==0, -1, original_labels1)
    labels2 = np.where(original_labels2==0, -1, original_labels2)

    tp = 0
    fp = 0
    fn = 0
    tn = 0 ## so far we are not calculating this, because it would be based on the background
    total = num_labels1


    detected_labels = []

    for i in range(1,len(centroids1)):
        x = int(round(centroids1[i][1]));
        y = int(round(centroids1[i][0]));
        label2 = labels2[x][y]
        label1 = labels1[x][y]
        if((label2>-1) and label2 not in detected_labels):
            detected_labels.append(label2)
            labels1 = np.where(labels1==label1, -1, labels1)
            labels2 = np.where(labels2==label2, -1, labels2)
            tp = tp + 1
        else:
            fn = fn + 1

    unique_labels1 = np.unique(labels1);
    unique_labels2 = np.unique(labels2);
    unique_labels1 = np.delete(unique_labels1, np.where(unique_labels1 == -1))
    unique_labels2 = np.delete(unique_labels2, np.where(unique_labels2 == -1))

    fp = len(unique_labels2)
    fn += len(unique_labels1)

    precision = tp/ (tp+fp);
    recall = tp / (tp+fn);
    f1 = 2 * ((precision*recall)/(precision+recall));

    print(precision, recall, f1);

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.subplot(132)
    plt.title('OpenCV {}'.format(num_labels1))
    plt.imshow(original_labels1, cmap='nipy_spectral')
    plt.axis('off')
    plt.subplot(133)
    plt.title('OpenCV {}'.format(num_labels2))
    plt.imshow(original_labels2, cmap='nipy_spectral')
    plt.axis('off')



    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
