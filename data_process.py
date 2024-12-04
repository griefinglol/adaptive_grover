from collections import Counter
import numpy as np

def process(path):
    with open(path,'r')as file:
        data = file.read().splitlines()

    # get list
    data = [int(value) for value in data]

    # get the dic
    counts = Counter(data)
    counts = dict(sorted(counts.items()))

    # generate distribution
    distribution = []
    total_number = len(data)
    for key, value in counts.items():
        distribution.append(value/total_number)
    distribution = np.array(distribution)

    return data, distribution

def linear_search(data):
    step_dic = dict.fromkeys(range(32), 0)
    for index in range(len(data)):
        key = data[index]
        if step_dic.get(key) == 0:
            step_dic[key] = index + 1
    step_list = list(step_dic.values())

    return step_list

if __name__ == "__main__":
    heavy_tail_path = "data/heavy_tail_samples.txt"
    normal_path = "data/normal_distribution_samples.txt"
    quniform_path = "data/quasi_uniform_samples.txt"
    uniform_path = "data/uniform_distribution.txt"

    path_list = [heavy_tail_path, normal_path, quniform_path, uniform_path]
    classic_step = []
    for path in path_list:
        data, _ = process(path)
        classic_step.append(linear_search(data))

    with open('output/classic_output.txt', 'w') as f:
        for item in classic_step:
            f.write(str(item) + '\n')