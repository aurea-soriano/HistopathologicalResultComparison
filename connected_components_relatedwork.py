from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import measure
from skimage import filters
import cv2
import os
import os.path
from os import path

def main():

    dir_path =  'quPath/tnbc/'   #'MonuSeg/'
    gt_path = dir_path+'gt/'
    results_path = dir_path+'results/'

    f = open("resultTNBCRW.txt", "a")
    f.write("image_id;strategy;precision;recall;f1;strategy;precision;recall;f1;strategy;precision;recall;f1")


    for root, dirs, files in os.walk(gt_path):
        for filename in files:
            print(filename)
            line = []

            fname1=gt_path + filename
            fname2=results_path + filename.split('.')[0]+" (1, 0, 0, 512, 512)_binary.png"

            if(path.exists(fname1) and path.exists(fname2)):

                line.append(fname1)
                img1 = Image.open(fname1).convert('L')
                img1 = np.asarray(img1)

                img2 = Image.open(fname2).convert('L')
                img2 = np.asarray(img2)



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





                # Threshold it so it becomes binary
                ret12, thresh2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)  # ensure binary
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
                line.append("qupath")
                line.append(precision)
                line.append(recall)
                line.append(f1)




                my_line = ';'.join(map(str, line))
                f.write(my_line+"\n")




    f.close()

if __name__ == "__main__":
    main()
