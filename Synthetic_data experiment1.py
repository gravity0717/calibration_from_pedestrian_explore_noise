from pprint import pprint
import argparse
import json 
import numpy as np 
import numpy as np 
import matplotlib.pyplot as plt 


def visualize_result(x_value:np.array,
                    y_value:dict,
                    title:str):
    methods = ["IQR",
                "RANSAC_IQR",
                "RANSAC_IQR_2",
                "Vanilla model",
                "ZSCORE",
                "RANSAC_ZSCORE",
                "RANSAC_ZSCORE_2"]

    linestyles =  [ '--', '-.', ':', '-', 'solid', 'dashed', 'dashdot', 'dotted']
    params = ["f",
              "theta",
              "phi",
              "height"]
   
    plt.title(title+" of models")
    plt.figure()

    for i, param in enumerate(params):

        # Extract first subplot to get legend
        if i == 0:
            ax1 = plt.subplot(2, 2, i+1)
        else: 
            plt.subplot(2, 2, i+1)
        
        for j , method in enumerate(methods):
            # plot graph
            plt.plot(x_value, y_value[method][param], linestyle = linestyles[j], label = method)

            # Configuration of axises
            plt.xlabel('number of line segments')
            plt.ylim([0,100])
            plt.ylabel(f"Error of {param}")
            
    # Adjust width spaces and horizontal space between subplots
    plt.subplots_adjust(wspace = 0.3, hspace = 0.5) 

    # Legend 
    handles, labels = ax1.get_legend_handles_labels()
    plt.figlegend(handles, labels, fontsize = 7, loc = "upper center", ncol = 4)

    # Save
    plt.savefig(f"result/{title}1.pdf")
    plt.savefig(f"result/{title}1.png")


if __name__ == "__main__":
    with open("metadata/exp_result_syn.json",'r') as f:
        x = json.load(f)
        x = x["median"]
    
    visualize_result(np.arange(2,100),x,"Synthetic data experiment")
    