import sys
sys.path.append("/media/grartem/B7DB5B36121B73AA/Projects/RobotCommandParser_service")
import yaml
import time
from RobotCommandParser.GappingUtils.GappingWrapper import GappingWrapper

CONFIG = {
             'use_gpu': False,
              'task_name': 'agrr',
              'bert_model': '/media/grartem/B7DB5B36121B73AA/Projects/RobotCommandParser_service/Models/Bert'
                            '/rubert_tiny2',
              'checkpoint_path': '/media/grartem/B7DB5B36121B73AA/Projects/RobotCommandParser_service/Models/Gapping'
                                 '/lower_tiny2/FriMay272022_model_f1_score=8130.pt',
              'max_seq_length': 128,
              'do_lower_case': False
}
"""
CONFIG = {
             'use_gpu': False,
              'task_name': 'agrr',
              'bert_model': '/media/grartem/B7DB5B36121B73AA/Projects/HuggingFace/models/bert-base-multilingual-cased',
              'checkpoint_path': '/media/grartem/B7DB5B36121B73AA/Projects/RobotCommandParser_service/Models/Gapping'
                                 '/lower_multi/SatJun182022_model_f1_score=0.9460.pt',
              'max_seq_length': 128,
              'do_lower_case': False
}
"""

gappingWrapper = GappingWrapper(CONFIG)
gappingWrapper.load_model()
start_time = time.time()
counter = 0
time_deltas = []
with open("/media/grartem/B7DB5B36121B73AA/Projects/RobotCommandParser_service/Data/gapping_test_syntagrus.csv", "r") as f:
#with open("/media/grartem/B7DB5B36121B73AA/Projects/RobotCommandParser_service/Data/gap_examples.txt", "r") as f:
    for line in f:
        short_start_time = time.time()
        prediction = gap_resolution_results = gappingWrapper.predict([line.strip()])
        time_deltas.append(time.time() - short_start_time)
        counter += 1
print(counter, "lines processed in {} seconds".format(time.time()-start_time))
avg_time = sum(time_deltas) / len(time_deltas)
print("AVG line time to process lines one by one: {}".format(avg_time))