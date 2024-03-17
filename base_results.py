import pandas as pd
from sklearn.metrics import confusion_matrix

def map_pred(x):
    if x[:4].lower() == 'true':
        return True

    if x[:5].lower() == 'false':
        return False

    if 'not needed' in x.lower():
        return False

    if 'not necessary' in x.lower():
        return False

    if 'not strictly needed' in x.lower():
        return False

    if 'not strictly necessary' in x.lower():
        return False

    if 'not necessarily needed' in x.lower():
        return False

    return None


if __name__ == '__main__':
    df = pd.read_csv('generated_data/preds.csv')

    df['pred_parsed'] = None
    df['pred_parsed'] = df['y_pred'].map(map_pred)

    df.to_csv('generated_data/preds_parsed.csv', index=False)

    print(df[df['pred_parsed'].isna()]['y_pred'].shape)

    df = df[(df['pred_parsed'] == True) | (df['pred_parsed'] == False)]
    correct = df['y_true'] == df['pred_parsed']

    print('\n\n\n\n')
    print(confusion_matrix(df['y_true'].astype(bool), df['pred_parsed'].astype(bool)))

    print('Accuracy: ', correct.sum() / df.shape[0])
