import torch

import pandas as pd

from ContentBasedLearning import ContentBasedLearning
from GraphBasedLearning import GraphBasedLearning

from config import train_file, test_file, PATH

from sklearn.metrics import recall_score, precision_score, f1_score, classification_report


train_path = '/data/akd_united_pseudo_data_1_1.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cbl = ContentBasedLearning(input_size=2048, device=device)

x_train, y_train, x_test, y_test = cbl.create_train_test_dataset(train_file=train_path, test_file=test_file)