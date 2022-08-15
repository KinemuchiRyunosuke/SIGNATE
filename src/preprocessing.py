import numpy as np
import pandas as pd
import tensorflow as tf
from bs4 import BeautifulSoup
import pickle

num_words = 1000


def main():
    csv_path = "data/raw/train.csv"
    df = pd.read_csv(csv_path)

    x = df['description'].to_numpy()
    y = df['jobflag'].to_numpy()

    # htmlタグを削除
    for i, text in enumerate(x):
        soup = BeautifulSoup(text, 'html.parser')
        x[i] = soup.get_text()

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=num_words,
        lower=True,
        oov_token='<UNK>'
    )
    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_sequences(x)

    x = pad_dataset(x, class_token=True)

    # One-Hot Encoding
    y = tf.keras.utils.to_categorical(y - 1, num_classes=4)

    # 前処理済みのデータを保存
    processed_path = "data/processed/processed_dataset.pickle"
    with open(processed_path, 'wb') as f:
        pickle.dump(x, f)
        pickle.dump(y, f)


def pad_dataset(sequences, class_token=False):
    """ paddingを行う

    paddingの値は0，<CLS>トークンは1とする．

    Arg:
        sequences: list of int
        class_token: Trueの場合，Transformerでクラスタリングを行う際に
            用いる<CLS>トークンを先頭に追加する．
    Return:
        ndarray: shape=(len(sequences), max_len)
            class_token=True の時は，shape=(len(sequences), max_len + 1)
    """
    if class_token:
        for i, seq in enumerate(sequences):
            sequences[i] = list(map(lambda x: x+1, seq))

    maxlen = max(map(lambda seq: len(seq), sequences))

    # shape=(len(sequences), max_len)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=maxlen, padding='post', value=0)

    if class_token:  # class_tokenを追加
        # class_id = 1
        cls_arr = np.ones((len(sequences), 1))     # shape=(len(sequences), 1)
        sequences = np.hstack([cls_arr, sequences]).astype('int64')

    return sequences

if __name__ == '__main__':
    main()
