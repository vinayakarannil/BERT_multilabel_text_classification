from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import argparse

from loss_util import focal_loss
from data_util import Dataset
from bert_util import Bert
from model import MultiLabelClassifier

parser = argparse.ArgumentParser(description='train multi label classification using bert encoding')

parser.add_argument('-train_data', type=str, default='data/train.csv',
                    help='path to the train.csv')
parser.add_argument('-test_data', type=str, default='data/test.csv',
                    help='path to the test.csv')
parser.add_argument('-ip', type=str, default='127.0.0.1',
                    help='ip of the server where bert server is running')

args = parser.parse_args()

############ get the train and val data splitted #########################
dataset = Dataset(args.train_data)
x_train, x_test, y_train, y_test = dataset.getDataset(test_size=0.2, random_state=0)

######## get bert encoding for train and validation datasets################
bc = Bert(args.ip)
x_train = bc.getBertEncoding(x_train)
x_test = bc.getBertEncoding(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
############################################################################


classifier = MultiLabelClassifier()
history, results = classifier.train(x_train,
                           y_train,
                           x_test,
                           y_test,
                           optimizer='adam',
                           loss=[focal_loss(alpha=.10, gamma=2)],
                           metrics='auc',
                           batch_size=128,
                           epochs=100
                           )

classifier.test_sample(x_train[:3], y_train[:3])

