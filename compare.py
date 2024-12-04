import matplotlib.pyplot as plt
import ast
import numpy as np

if __name__ == "__main__":
    classic_out_path = "output/classic_output.txt"
    quantum_out_path = "output/quantum_output.txt"

    with open(classic_out_path,'r')as file:
        classic_step = file.read().splitlines()

    with open(quantum_out_path,'r')as file:
        quantum_step = file.read().splitlines()
    
    classic_step = [ast.literal_eval(string) for string in classic_step] 
    quantum_step = [ast.literal_eval(string) for string in quantum_step] 
    distribution_list = ["heavy_tail", "noraml", "quasi_uniform", "unifrom"]

    for index in range(len(distribution_list)):
        # plot the data
        plt.plot(np.log(classic_step[index]), label = "classic", color = "blue")
        plt.plot(np.log(quantum_step[index]), label = "quantum", color = "red")

        plt.title(distribution_list[index])
        plt.xlabel("Index")
        plt.ylabel("Values")

        plt.legend()

        save_path = "output/" + distribution_list[index] + ".jpg"
        plt.savefig(save_path, dpi = 300)

        plt.close()