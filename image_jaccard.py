from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import jaccard_similarity_score
import os
import os.path


def main():
    gt_path_name = "/Users/aurea/Documents/CBMS2020/cbms-2020/TNBC_NucleiSegmentation/GT_04/"
    results_path_name = "/Users/aurea/Documents/CBMS2020/cbms-2020/TNBC_NucleiSegmentation/Slide_04/"


    dir_files = os.listdir(gt_path_name)
    gt_list = []
    results_list = []

    for file in dir_files:
        if os.path.isfile(gt_path_name+file.split(".")[0]+".png") and os.path.isfile(results_path_name+file.split(".")[0]+"_result.png"):
            gt_list.append(gt_path_name+file.split(".")[0]+".png");
            results_list.append(results_path_name+file.split(".")[0]+"_result.png");
        else:
            print ("File not exist")


    for i in range (0, len(gt_list)):
        gt_image = Image.open(gt_list[i]).convert('LA')
        result_image = Image.open(results_list[i]).convert('LA')

        # make a 1-dimensional view of arr
        gt_vector =  np.ravel(np.array(gt_image))
        gt_vector = np.array(np.where(gt_vector==255, 1, gt_vector))

        result_vector =  np.ravel(np.array(result_image))
        result_vector = np.array(np.where(result_vector==255, 1, result_vector))

        jaccard_result = jaccard_similarity_score(gt_vector, result_vector)

        print(jaccard_result)




if __name__ == "__main__":
    main()
