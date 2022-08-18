import os
from numpy import result_type
import pandas as pd

def main():
    dirname = os.path.dirname(__file__)
    results_dirname = os.path.join(dirname, 'results/')

    unavailable_results = []
    df = pd.DataFrame(columns=['experiment_id', ' method ', ' base_learner ', ' task_id ', ' fold ', ' test_accuracy', ' train_accuracy', ' ensemble_size', ' duration'])

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

    return df

def parse_result_string(s):
    s = s.replace("experiment id: ", "")
    experiment_id, s = s.split(", method: ")
    method, s = s.split(", base_learner: ")
    base_learner, s = s.split(" task: ")
    task_id, s = s.split(", fold: ")
    fold, s = s.split(", accuracy: ")
    test_accuracy, s = s.split(", train_accuracy: ")
    train_accuracy, s = s.split(", ensemble_size: ")
    ensemble_size, s = s.split(", duration: ")
    duration, s = s.split(", other_results")

    return [experiment_id, method , base_learner , task_id , fold , test_accuracy, train_accuracy, ensemble_size, duration]


main()
#marcel die task ids zukommen lassen

