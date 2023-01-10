import os
from re import L
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random


def results_to_df(results_folder):
    dirname = os.path.dirname(__file__)
    results_dirname = os.path.join(dirname, results_folder)

    unavailable_results = []
    df = pd.DataFrame(columns=['experiment_id', 'method', 'base_learner', 'task_id', 'fold', 'test_accuracy', 'train_accuracy', 'ensemble_size', 'duration'])

    for i in range(400):
        next_result_path = os.path.join(results_dirname, str(i)+"/results.txt")
        next_result_path = next_result_path.replace("\\", "/")
        try:
            with open(next_result_path, "r") as file:
                result_str = file.readlines()[-1]
                df.loc[len(df)] = parse_result_string(result_str)
            

        except Exception as e:
            print("Somthing went wrong - probably " + next_result_path +" is missing")
            print(e)
            unavailable_results.append(i)
    print(unavailable_results)
    with open(results_dirname + "/results.csv", "w") as file:
        df.to_csv(file)

    compare_results(df)
    return df

def parse_result_string(s):
    s = s.replace("experiment id: ", "")
    experiment_id, s = s.split(", method: ")
    method, s = s.split(", base_learner: ")
    base_learner, s = s.split(" task: ")
    task_id, s = s.split(", fold: ")
    fold, s = s.split(", accuracy: ")
    test_accuracy, s = s.split(", train_accuracy: ")
    test_accuracy = float(test_accuracy)
    train_accuracy, s = s.split(", ensemble_size: ")
    train_accuracy = float(train_accuracy)
    ensemble_size, s = s.split(", duration: ")
    ensemble_size = int(ensemble_size)
    duration, s = s.split(", other_results")
    duration = float(duration)

    return [experiment_id, method , base_learner , task_id , fold , test_accuracy, train_accuracy, ensemble_size, duration]


def compare_results(df):
    mean_accuracies = df.loc[:, ["method", "base_learner", "task_id", "fold", "test_accuracy"]].groupby(["method", "base_learner", "task_id"]).mean().loc[:, "test_accuracy"]
    task_ids = set(mean_accuracies.index.get_level_values(2))
    base_learners = set(mean_accuracies.index.get_level_values(1))
    plt_number = 0
    fig, axs = plt.subplots(1,len(base_learners))
    for base_learner in base_learners:
        ndea = [] 
        conda = []
        for task_id in task_ids:
            x = ("ndea", base_learner, task_id)
            y = ("conda", base_learner, task_id)
            if(x in mean_accuracies.index and y in mean_accuracies.index):
                ndea.append(mean_accuracies.loc[x])
                conda.append(mean_accuracies.loc[y])
        axs[plt_number].set_title(base_learner)
        axs[plt_number].plot(ndea, conda, "ro")
        axs[plt_number].plot([0.4,1], [0.4,1], "b-")
        axs[plt_number].set(xlabel = "ndea", ylabel = "conda")
        plt_number += 1
    plt.tight_layout()
    plt.show()


def compare_time(df):
    base_learners = df["base_learner"].unique()
    task_ids = df["task_id"].unique()
    labels = list(map(dataset_name, task_ids))
    plt_number = 0
    fig, axs = plt.subplots(len(base_learners))
    for base_learner in base_learners:
        axs[plt_number].set_title(base_learner)
        axs[plt_number].set_xscale("log")
        conda_data = []
        ndea_data = []
        for task_id in task_ids:
            conda_task_data = df.loc[(df["base_learner"] == base_learner) & (df["task_id"] == task_id) & (df["method"] == "conda")]["duration"]
            ndea_task_data = df.loc[(df["base_learner"] == base_learner) & (df["task_id"] == task_id) & (df["method"] == "ndea")]["duration"]
            conda_data.append(conda_task_data)
            ndea_data.append(ndea_task_data)
        bp_conda = axs[plt_number].boxplot(conda_data, vert = False, showfliers=False, patch_artist=True, labels = labels)
        bp_ndea = axs[plt_number].boxplot(ndea_data, vert = False, showfliers=False, patch_artist=True, labels = labels)
        for patch in bp_conda['boxes']:
            patch.set_facecolor("green")
        for patch in bp_ndea['boxes']:
            patch.set_facecolor("pink")
        plt_number+=1
    plt.tight_layout()
    plt.show()

def compare_time_bar(df):
    base_learners = df["base_learner"].unique()
    task_ids = df["task_id"].unique()
    labels = list(map(dataset_name, task_ids))
    plt_number = 0
    fig, axs = plt.subplots(1, len(base_learners))
    bar_conda = None
    bar_ndea = None
    for base_learner in base_learners:
        conda_data = []
        ndea_data = []
        for task_id in task_ids:
            axs[plt_number].set_title(base_learner)
            conda_task_data = df.loc[(df["base_learner"] == base_learner) & (df["task_id"] == task_id) & (df["method"] == "conda")]["duration"].mean()
            ndea_task_data = df.loc[(df["base_learner"] == base_learner) & (df["task_id"] == task_id) & (df["method"] == "ndea")]["duration"].mean()
            conda_data.append(conda_task_data)
            ndea_data.append(ndea_task_data)
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        bar_conda = axs[plt_number].barh(x-width/2, conda_data, width, color = "green", label = "CONDA")
        bar_ndea = axs[plt_number].barh(x+width/2, ndea_data, width, color = "pink", label = "NDEA")
        axs[plt_number].set(xlabel = "Runtime")
        axs[plt_number].set_yticks(x, labels)
        plt_number+=1

    fig.legend([bar_conda, bar_ndea], labels = ["CONDA", "NDEA"], loc="upper right")
 

    plt.tight_layout()
    plt.show()

def compare_sizes(df):
    base_learners = df["base_learner"].unique()
    task_ids = df["task_id"].unique()
    result = pd.DataFrame()
    mean_sizes = df.loc[:, ["method", "base_learner", "task_id", "fold", "ensemble_size"]].groupby(["method", "base_learner", "task_id"]).mean().loc[:, "ensemble_size"]
    return mean_sizes

def compare_overfitting(df):
    base_learners = df["base_learner"].unique()
    task_ids = df["task_id"].unique()
    result = pd.DataFrame()
    mean_acc = df.loc[:, ["method", "base_learner", "task_id", "fold", "train_accuracy", "test_accuracy"]].groupby(["method", "base_learner", "task_id"]).mean().loc[:, ["train_accuracy", "test_accuracy"]]
    return mean_acc

def compare_results_over_all_tasks(df):
    mean_accuracies = df.loc[:, ["method", "base_learner", "task_id", "fold", "test_accuracy"]].groupby(["method", "base_learner", "task_id"]).mean().loc[:, "test_accuracy"]
    task_ids = set(mean_accuracies.index.get_level_values(2))
    base_learners = set(mean_accuracies.index.get_level_values(1))
    for base_learner in base_learners:
        ndea = [] 
        conda = []
        for task_id in task_ids:
            x = ("ndea", base_learner, task_id)
            y = ("conda", base_learner, task_id)
            if(x in mean_accuracies.index and y in mean_accuracies.index):
                ndea.append(mean_accuracies.loc[x])
                conda.append(mean_accuracies.loc[y])
        plt.title(base_learner)
        plt.plot(ndea, conda, "ro")
        plt.plot([0.4,1], [0.4,1], "b-")
        plt.xlabel("ndea")
        plt.ylabel("conda")
        plt.tight_layout()
        plt.show()

def test():
    dirname = os.path.dirname(__file__)
    results_path = os.path.join(dirname, 'results/results.csv')
    df = pd.read_csv(results_path)
    return df

def dataset_name(task_id):
    dict = {
        "anneal": 2,
        "audiology": 7,
        "autos": 9,
        "glass": 40,
        "led24": 146204,
        "mfeat-morphological": 18,
        "semeion": 9964,
        "soybean":  41,
        "vowel": 3022,
        "wine-quality-white": 145681,
    }
    reverse_dict = {}
    for key in dict.keys():
        reverse_dict[dict[key]] = key
    return reverse_dict[task_id]

def structure_example():
    class1 = [(1+(random.random()-0.5)/2, 1+(random.random()-0.5)/2 ) for i in range(10)]
    class2 = [(2+(random.random()-0.5)/2, 1+(random.random()-0.5)/2 ) for i in range(10)]
    class3 = [(1+(random.random()-0.5)/2, 2+(random.random()-0.5)/2 ) for i in range(10)]
    class4 = [(2+(random.random()-0.5)/2, 2+(random.random()-0.5)/2 ) for i in range(10)]

    fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
    #easy decision boundry
    axs[0].plot(*list(zip(*class1)), "ro")
    axs[0].plot(*list(zip(*class2)), "gs")
    axs[0].plot(*list(zip(*class3)), "r^")
    axs[0].plot(*list(zip(*class4)), "gD")
    axs[0].plot([1.4,1.5], [0.8,2.2], "k-")
    #axs[0].set_aspect('equal', 'box')    
    #axs[0].set(adjustable = "box-forced", aspect = "equal")

    axs[1].plot(*list(zip(*class1)), "go")
    axs[1].plot(*list(zip(*class2)), "rs")
    axs[1].plot(*list(zip(*class3)), "r^")
    axs[1].plot(*list(zip(*class4)), "gD")
    axs[1].plot([0.75,2.25], [1.4,1.6], "k-")
    #axs[1].set_aspect('equal', 'box')  
    #axs[1].set(adjustable = "box-forced", aspect = "equal")
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='$c_1$', markerfacecolor='k', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='$c_2$', markerfacecolor='k', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='$c_3$', markerfacecolor='k', markersize=10),
        Line2D([0], [0], marker='D', color='w', label='$c_4$', markerfacecolor='k', markersize=10)]
    fig.legend(handles=legend_elements, loc='upper center', ncol = 4) #, bbox_to_anchor=[0, 0.9]
    plt.show()



if __name__ == "__main__":
    compare_results(results_to_df("results-fixed/"))