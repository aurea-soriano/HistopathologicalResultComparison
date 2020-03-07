from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import measure
from skimage import filters
import cv2
import os

def main():

    dir_path = 'MoNuSeg/'
    gt_path = dir_path+'gt/'
    r_moments_path = dir_path+'results/moments/'
    r_otsu_path = dir_path+'results/otsu/'
    r_triangle_path = dir_path+'results/triangle/'
    
    if os.path.exists("resultMonuSeg.txt"):
        os.remove("resultMonuSeg.txt")

    f = open("resultMonuSeg.txt", "a")
    f.write("image_id;strategy;precision;recall;f1;strategy;precision;recall;f1;strategy;precision;recall;f1\n")
    
    
    
    for root, dirs, files in os.walk(gt_path):
        for filename in files:
            print(filename)
            line = []

            fname1=gt_path + filename
            line.append(fname1)

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



            fname2=r_moments_path + filename.split('.')[0]+"_result_label.png"
            img2 = Image.open(fname2).convert('L')
            img2 = np.asarray(img2)

            # The second cell is the label matrix
            original_labels2 = img2



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
            line.append("moments")
            line.append(precision)
            line.append(recall)
            line.append(f1)



            fname2=r_otsu_path + filename.split('.')[0]+"_result_label.png"
            img2 = Image.open(fname2).convert('L')
            img2 = np.asarray(img2)

            # The second cell is the label matrix
            original_labels2 = img2



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
            line.append("otsu")
            line.append(precision)
            line.append(recall)
            line.append(f1)



            fname2=r_triangle_path + filename.split('.')[0]+"_result_label.png"
            img2 = Image.open(fname2).convert('L')
            img2 = np.asarray(img2)

            # The second cell is the label matrix
            original_labels2 = img2



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
            line.append("triangle")
            line.append(precision)
            line.append(recall)
            line.append(f1)

            my_line = ';'.join(map(str, line))
            f.write(my_line+"\n")

            '''plt.figure(figsize=(9, 3.5))
            plt.subplot(131)
            plt.title('Original Image')
            plt.imshow(img1, cmap='gray')
            plt.axis('off')
            plt.subplot(132)
            plt.title('OpenCV {}'.format(num_labels1))
            plt.imshow(original_labels1, cmap='nipy_spectral')
            plt.axis('off')
            plt.subplot(133)
            plt.title('OpenCV')
            plt.imshow(original_labels2, cmap='nipy_spectral')
            plt.axis('off')



            plt.tight_layout()
            plt.show()'''


    f.close()

if __name__ == "__main__":
    main()
