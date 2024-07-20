from tqdm import tqdm
import tensorflow as tf

tf.get_logger().setLevel('ERROR')  # only show error messages

import pandas as pd

from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SEED, EMBEDDINGS_PATH

from utils import *

import warnings

warnings.filterwarnings('ignore')


class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)  # Fully connected layer 1
        self.dropout1 = nn.Dropout(0.8)  # Dropout layer with 20% probability
        self.fc2 = nn.Linear(512, 256)  # Fully connected layer 2
        self.dropout2 = nn.Dropout(0.8)  # Dropout layer with 20% probability
        self.fc3 = nn.Linear(256, 1)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first fully connected layer
        x = self.dropout1(x)  # Apply dropout to the output of the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to the second fully connected layer
        x = self.dropout2(x)  # Apply dropout to the output of the second layer
        x = torch.sigmoid(self.fc3(x))  # Apply sigmoid activation to the output layer for binary classification
        return x


class ContentBasedLearning:
    def __init__(self, input_size, device):
        # Assuming x_train is your input tensor
        self.input_size = input_size  # This should be 2048 (1024 * 2) based on your previous code

        # Define loss function and optimizer
        self.criterion = nn.BCELoss()  # Binary Cross Entropy Loss

        # Load the embedding dataframes
        self.embedding_dfs = {
            'al_cpl': pd.read_csv(EMBEDDINGS_PATH + r'\al_cpl_embeddings_mistral.csv', index_col='Unnamed: 0').to_dict(orient='index'),
            'drive': pd.read_csv(EMBEDDINGS_PATH + r'\drive_embeddings_mistral.csv', index_col='Unnamed: 0').to_dict(orient='index'),
            'mooc': pd.read_csv(EMBEDDINGS_PATH + r'\mooc_embeddings_mistral.csv', index_col='Unnamed: 0').to_dict(orient='index')
        }

        self.embeddings_dict = dict()
        self.embeddings_dict['al_cpl'] = dict()
        self.embeddings_dict['drive'] = dict()
        self.embeddings_dict['mooc'] = dict()

        for i, (key, value) in enumerate(self.embedding_dfs['al_cpl'].items()):
            self.embeddings_dict['al_cpl'].update({i: list(value.values())})
        for i, (key, value) in enumerate(self.embedding_dfs['mooc'].items()):
            self.embeddings_dict['mooc'].update({i: list(value.values())})
        for i, (key, value) in enumerate(self.embedding_dfs['drive'].items()):
            self.embeddings_dict['drive'].update({i: list(value.values())})

    # Function to get embedding for a concept
    def get_embedding(self, file, concept_ind):
        # embedding = self.embedding_dfs[file].loc[concept_ind].iloc[1:].values
        # return embedding
        return self.embeddings_dict[file][concept_ind]

    # Function to process a dataset
    def process_dataset(self, df):
        x_list = []
        y_list = []

        for _, row in df.iterrows():
            fileA = row['file']
            fileB = row['fileB']

            # Get embeddings for conceptA and conceptB
            embedding_a = self.get_embedding(fileA, row['conceptA_ind'])
            embedding_b = self.get_embedding(fileB, row['conceptB_ind'])

            # Combine the embeddings
            combined_features = np.concatenate([embedding_a, embedding_b])

            x_list.append(combined_features)
            y_list.append(row['isPrerequisite'])

        return x_list, y_list

    def create_train_test_dataset(self, train_file, test_file):
        # Load the train, test set
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)

        # Process train and test sets
        x_train_list, y_train_list = self.process_dataset(train)
        x_test_list, y_test_list = self.process_dataset(test)

        # Convert to PyTorch tensors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x_train = torch.tensor(x_train_list, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train_list, dtype=torch.long).to(self.device)

        x_test = torch.tensor(x_test_list, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test_list, dtype=torch.long).to(self.device)

        print(f"x_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"x_test shape: {x_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        return x_train, y_train, x_test, y_test

    def train_model(self, x_train, y_train, x_test, y_test, num_epochs):
        # Initialize the model
        self.model = BinaryClassifier(self.input_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            self.model.train()
            outputs = self.model(x_train)
            loss = self.criterion(outputs, y_train.float().unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.model.eval()  # Set the model to evaluation mode
                train_preds = (outputs >= 0.5).float()  # Convert probabilities to binary predictions
                train_acc = accuracy_score(y_train.cpu().numpy(), train_preds.cpu().numpy())
                train_f1 = f1_score(y_train.cpu().numpy(), train_preds.cpu().numpy())

            with torch.no_grad():
                test_outputs = self.model(x_test)
                test_preds = (test_outputs >= 0.5).float()  # Convert probabilities to binary predictions
                test_acc = accuracy_score(y_test.cpu().numpy(), test_preds.cpu().numpy())
                test_f1 = f1_score(y_test.cpu().numpy(), test_preds.cpu().numpy())

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}')

    def generate_all_predictions(self, train):
        self.model.eval()  # Set the model to evaluation mode

        users, items, preds = [], [], []
        item = list(train["conceptB"].unique())

        a_info = dict()
        for c in train.iterrows():
            a = c[1]
            a_info[a['conceptA']] = (a['file'], a['conceptA_ind'])

        b_info = dict()
        for c in train.iterrows():
            a = c[1]
            b_info[a['conceptB']] = (a['fileB'], a['conceptB_ind'])

        for user in tqdm(train["conceptA"].unique()):
            used_items = set(train.loc[train['conceptA'] == user, 'conceptB'])
            user_items = []

            for i in item:
                if i in used_items:
                    continue
                user_items.append([user, i])
                users.append(user)
                items.append(i)

            # Create embeddings for user_items
            user_item_embeddings = []
            for ui in user_items:
                ui_0 = a_info[ui[0]]
                ui_1 = b_info[ui[1]]

                embedding_a = self.get_embedding(ui_0[0], ui_0[1])
                embedding_b = self.get_embedding(ui_1[0], ui_1[1])

                combined_features = np.concatenate([embedding_a, embedding_b])
                user_item_embeddings.append(combined_features)

            # Convert to tensor and move to device
            user_item_tensor = torch.tensor(user_item_embeddings, dtype=torch.float32).to(self.device)

            # Get predictions
            with torch.no_grad():
                batch_preds = self.model(user_item_tensor).squeeze().cpu().numpy()

            preds.extend(list(batch_preds))

        pseudo_predictions = pd.DataFrame(data={"conceptA": users, "conceptB": items, "isPrerequisite_pred": preds})

        pseudo_predictions.sort_values(by=['conceptA', 'isPrerequisite_pred'], ascending=[True, False], inplace=True)

        return pseudo_predictions

    def create_new_row(self, row, train, is_pos):
        # Find the row in train for conceptA
        conceptA_row = train[train['conceptA'] == row['conceptA']].iloc[0]

        # Find the row in train for conceptB
        conceptB_row = train[train['conceptB'] == row['conceptB']].iloc[0]

        # Create a new row with all columns from conceptA_row
        new_row = conceptA_row.copy()

        new_row['conceptA'] = row['conceptA']
        new_row['conceptB'] = row['conceptB']
        new_row['isPrerequisite'] = is_pos

        new_row['dataset'] = 'pseudo'
        new_row['file'] = conceptA_row['file']
        new_row['fileB'] = conceptB_row['file']

        new_row['conceptA_ind'] = conceptB_row['conceptA_ind']
        new_row['conceptB_ind'] = conceptB_row['conceptB_ind']

        return new_row

    def generate_pseudo_data(self, train_file):
        train = pd.read_csv(train_file)

        pseudo_predictions = self.generate_all_predictions(train)

        vs = train.groupby('conceptA').agg('sum')['isPrerequisite'].reset_index()

        np.random.seed(SEED)  # Use a fixed seed for reproducibility

        vs['k'] = (vs['isPrerequisite']).apply(biased_coin_toss)

        print(f"Original positives: {vs['isPrerequisite'].sum()}, Pseudo positives: {vs['k'].sum()}")

        pseudo_predictions = pseudo_predictions.merge(vs[['conceptA', 'k']], on='conceptA', how='left')

        pseudo_pos = pseudo_predictions.groupby('conceptA').apply(lambda x: x.head(int(x['k'].values[0]))).reset_index(
            drop=True)
        pseudo_neg = pseudo_predictions.groupby('conceptA').apply(lambda x: x.tail(int(x['k'].values[0]))).reset_index(
            drop=True)

        # Create new rows for positive pseudo-labels
        new_pos_rows = [self.create_new_row(row, train, 1) for _, row in pseudo_pos.iterrows()]

        # Create new rows for negative pseudo-labels
        new_neg_rows = [self.create_new_row(row, train, 0) for _, row in pseudo_neg.iterrows()]

        pseudo_data = pd.DataFrame(new_pos_rows + new_neg_rows)

        return pseudo_data
