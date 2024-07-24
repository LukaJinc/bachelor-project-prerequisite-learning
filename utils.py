import numpy as np
import pandas as pd
from config import COIN_TOSS_PROB


def biased_coin_toss(n, p=COIN_TOSS_PROB):
    results = np.random.binomial(1, p, n)
    count_of_ones = np.sum(results)
    return count_of_ones


def create_new_row(row, train, is_pos):
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
    new_row['file'] = conceptA_row['fileA']
    new_row['fileB'] = conceptB_row['fileB']

    new_row['conceptA_ind'] = conceptA_row['conceptA_ind']
    new_row['conceptB_ind'] = conceptB_row['conceptB_ind']

    return new_row


def unite_pos_neg(pseudo_predictions, vs, train):
    pseudo_predictions = pseudo_predictions.merge(vs[['conceptA', 'k']], on='conceptA', how='left')

    pseudo_pos = pseudo_predictions.groupby('conceptA').apply(lambda x: x.head(int(x['k'].values[0]))).reset_index(
        drop=True)
    pseudo_neg = pseudo_predictions.groupby('conceptA').apply(lambda x: x.tail(int(x['k'].values[0]))).reset_index(
        drop=True)

    # Create new rows for positive pseudo-labels
    new_pos_rows = [create_new_row(row, train, 1) for _, row in pseudo_pos.iterrows()]

    # Create new rows for negative pseudo-labels
    new_neg_rows = [create_new_row(row, train, 0) for _, row in pseudo_neg.iterrows()]

    pseudo_data = pd.DataFrame(new_pos_rows + new_neg_rows)

    return pseudo_data
