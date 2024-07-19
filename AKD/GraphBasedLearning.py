import pandas as pd

from ncf.ncf import NCF
from ncf.dataset import Dataset as NCFDataset

from config import SEED, ENCODED_SPLIT_PATH, NCF_THRESHOLD

from utils import *

import warnings

warnings.filterwarnings('ignore')


class GraphBasedLearning:
    def __init__(self, col_user, col_item):
        self.col_user = col_user
        self.col_item = col_item

    def train(self, train_file):
        data = NCFDataset(train_file=train_file, seed=SEED, col_user='conceptA', col_item='conceptB')

        self.model = NCF(
            n_users=data.n_users,
            n_items=data.n_items,
            model_type="NeuMF",
            n_factors=16,
            layer_sizes=[8, 4],
            n_epochs=20,
            batch_size=256,
            learning_rate=0.001,
            verbose=20,
            seed=SEED
        )

        self.model.fit(data)

    def predict(self):
        df = pd.read_csv(ENCODED_SPLIT_PATH)
        predictions = [[row.conceptA, row.conceptB, self.model.predict(row.conceptA, row.conceptB)]
                       for (_, row) in df.iterrows()]

        predictions = pd.DataFrame(predictions, columns=['conceptA', 'conceptB', 'isPrerequisite_pred'])

        predictions['isPrerequisite'] = df['isPrerequisite']
        predictions['dataset'] = df['dataset']
        predictions['_split_set'] = df['_split_set']
        sorted_predictions = predictions.sort_values(by='isPrerequisite_pred', ascending=False)
        sorted_predictions['pred'] = (sorted_predictions['isPrerequisite_pred'] >= NCF_THRESHOLD).astype(int)

        return sorted_predictions

    def generate_pseudo_data(self, train_file):
        train = pd.read_csv(train_file)

        users, items, preds = [], [], []
        item = list(train["conceptB"].unique())

        for user in train["conceptA"].unique():
            itemsA = list(train.loc[train['conceptA'] == user, "conceptB"].unique())

            item = list(train.loc[~(train['conceptB'].isin(itemsA)), "conceptB"].unique())

            user = [user] * len(item)
            users.extend(user)
            items.extend(item)
            preds.extend(list(self.model.predict(user, item, is_list=True)))

        pseudo_predictions = pd.DataFrame(data={"conceptA": users, "conceptB": items, "isPrerequisite_pred": preds})
        pseudo_predictions.sort_values(by=['conceptA', 'isPrerequisite_pred'], ascending=[True, False], inplace=True)

        vs = train.groupby('conceptA').agg('sum')['isPrerequisite'].reset_index()

        np.random.seed(SEED)

        vs['k'] = (vs['isPrerequisite']).apply(biased_coin_toss)

        vs['isPrerequisite'].sum(), vs['k'].sum()

        pseudo_predictions = pseudo_predictions.merge(vs[['conceptA', 'k']], on='conceptA', how='left')

        pseudo_pos = pseudo_predictions.groupby('conceptA').apply(lambda x: x.head(int(x['k'].values[0]))).reset_index(
            drop=True)
        pseudo_neg = pseudo_predictions.groupby('conceptA').apply(lambda x: x.tail(int(x['k'].values[0]))).reset_index(
            drop=True)

        return pseudo_pos, pseudo_neg
