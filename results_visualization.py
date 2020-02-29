import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def main():

    sns.set(style="ticks")
    sns.set(style="darkgrid")
    result_file = open("results.txt", "r")
    result_values = result_file.read().split('\n')
    result_values = result_values[:-1]
    result_values =  np.array(result_values, dtype=np.float32)
    print(result_values)

    # get x and y vectors
    y = result_values;
    mean = np.mean(result_values)
    y_mean = [mean] * len(result_values)
    #print(y_mean)
    x = list(range(1, len(result_values)+1));
    #print(x)
    #print(y)

    # calculate polynomial
    z = np.polyfit(x, y, 3)
    f = np.poly1d(z)

    # calculate new x's and y's
    x_new = np.linspace(x[0], x[-1], 50)
    y_new = f(x_new)

    plt.plot(x,y,marker='o', color=('#0077b3'), markersize=8)
    plt.plot(x, y, color=('#ffad33'), linewidth=2)
    plt.xlim([x[0]-1, x[-1] + 1 ])
    plt.ylim(0,1.2)

    plt.plot(x,y_mean, color=('#009900'), linestyle='dashed')

    plt.show()

if __name__ == "__main__":
    main()
