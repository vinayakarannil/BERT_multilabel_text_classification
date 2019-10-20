import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:

    def __init__(self, file_path):
        self.file_path = file_path

    def getDataset(self, test_size=0.2, random_state=0):
        data = pd.read_csv(self.file_path)
        data = data[(data['comment_text'] != '')]
        x = data['comment_text']
        y = data.loc[:, ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
        return train_test_split(x, y, test_size=0.2, random_state=0)
