from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import jaccard_similarity_score
import os
import os.path


def main():
    
    datasetPath = "/home/oscar/src/HistopathologicalResultComparison/quPath/tnbc";
    
    gt_path_name = datasetPath+"/gt"
    #thresholdMethod = "triangle"
    #results_path_name = datasetPath+"/results/"thresholdMethod
    results_path_name = datasetPath+"/results"
    dataPath = datasetPath+"/data"
    
        
    for file in os.listdir(dataPath):
        
        imageName = file.split( "/" )[-1].split(".")[0]
        #print(imageName)
        
        try:
            gt_image = Image.open(gt_path_name+"/"+imageName+".png"  ).convert('LA')
            result_image = Image.open(results_path_name+"/"+imageName+" (1, 0, 0, 512, 512)_binary.png").convert('LA')
        except:    
            #print("shit")
            continue
        
        # make a 1-dimensional view of arr
        gt_vector =  np.ravel(np.array(gt_image))
        gt_vector = np.array(np.where(gt_vector==255, 1, gt_vector))

        result_vector =  np.ravel(np.array(result_image))
        result_vector = np.array(np.where(result_vector==255, 1, result_vector))

        jaccard_result = jaccard_similarity_score(gt_vector, result_vector)

        #print(thresholdMethod + ", "+imageName+", "+str(jaccard_result))
        print(imageName+", "+str(jaccard_result))
        
        
        
        
        #if os.path.isfile(gt_path_name+file.split(".")[0]+".png") and os.path.isfile(results_path_name+file.split(".")[0]+"_result_binary.png"):
            #gt_list.append(gt_path_name+file.split(".")[0]+".png");
            #results_list.append(results_path_name+file.split(".")[0]+"_result_binary.png");
        #else:
            #print ("File not exist "+results_path_name+file.split(".")[0]+"_result_binary.png")


#    for i in range (0, len(gt_list)):
#        gt_image = Image.open(gt_list[i]).convert('LA')
#        result_image = Image.open(results_list[i]).convert('LA')
#
#        # make a 1-dimensional view of arr
#        gt_vector =  np.ravel(np.array(gt_image))
#        gt_vector = np.array(np.where(gt_vector==255, 1, gt_vector))
#
#        result_vector =  np.ravel(np.array(result_image))
#        result_vector = np.array(np.where(result_vector==255, 1, result_vector))
#
#        jaccard_result = jaccard_similarity_score(gt_vector, result_vector)
#
#        print(jaccard_result)




if __name__ == "__main__":
    main()
