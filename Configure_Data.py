import _pickle as pickle
import os

"""
Groups all data into a single pickle file and adjusts some of the rewards
"""

EXPERT_DIR = "Expert Data"
files = os.listdir(EXPERT_DIR)
file_count = len(files)
rewards = []
full_expert_data = []
for i in range(file_count):
    with open(EXPERT_DIR + "/Data_{}.pkl".format(i), 'rb') as handle:
        batch = pickle.load(handle)
    for j in range(len(batch)):
        batch[j][2] = batch[j][2] / 5
        if batch[j][3]:
            batch[j][2] = batch[j][2] + 5
        rewards.append(batch[j][2])
        full_expert_data.append(batch[j])
with open('Complete_Expert_Data.pkl', 'wb') as handle:
    pickle.dump(full_expert_data, handle)
