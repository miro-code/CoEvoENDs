import os
from py_experimenter.result_processor import ResultProcessor
from ensemble_construction import single_experiment
from py_experimenter.experimenter import PyExperimenter


def run_experiment(parameters: dict, result_processor: ResultProcessor, custom_fields:dict):
    # Extracting given parameters
    _, _, _, _, _, accuracy, train_accuracy, ensemble_size, duration, _ = single_experiment(parameters["method"], parameters["task_id"], parameters["fold_id"], parameters["base_learner"])
    # Do some stuff here

    # Write intermediate results to database    
    resultfields = {
        'accuracy': accuracy, 
        'train_accuracy': train_accuracy,
        'ensemble_size': ensemble_size,
        "duration" : duration
        }
    result_processor.process_results(resultfields)


if __name__ == "__main__":

    experimenter = PyExperimenter("config/experiments.cfg" )
    experimenter.fill_table_from_config()
    experimenter.execute(run_experiment, max_experiments=1, random_order=True)
    print(test)