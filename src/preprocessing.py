from optparse import OptionValueError
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import gensim

from bs4 import BeautifulSoup
from transformers import AutoTokenizer

num_words = 1000

csv_path = "data/raw/train.csv"
pretrained_model_name_or_path = 'bert-base-uncased'
processed_path = "data/processed/processed_dataset.pickle"
word_vectors_path = "data/wiki-news-300d-1M.vec"
embedding_path = "data/embedding.pickle"


def main():
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
    )

    vocab = {'<PAD>':0, '<UNK>':1, '<CLS>':2}
    tokenizer.word_index = vocab

    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_sequences(x)

    maxlen = max(list(map(lambda seq: len(seq), x)))
    x = tf.keras.preprocessing.sequence.pad_sequences(
            x, maxlen=maxlen, padding='post', value=0)

    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(word_vectors_path)
    word_vectors = filter_embeddings(word_vectors, tokenizer.word_index, num_words)

    with open(embedding_path, 'wb') as f:
        pickle.dump(word_vectors, f)

    # One-Hot Encoding
    y = tf.keras.utils.to_categorical(y - 1, num_classes=4)

    # 前処理済みのデータを保存
    with open(processed_path, 'wb') as f:
        pickle.dump(x, f)
        pickle.dump(y, f)


def convert_examples_to_features(x, maxlen, tokenizer, num_words):
    """文章をTransformerに入力可能な状態に変更する

    Args:
        x(list of str): 文章を格納したリスト
        maxlen(int): 最大の単語数

    Returns:
        list of str

    """
    seqs = []

    for words in x:
        tokens = [tokenizer.cls_token]
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        seqs.append(input_ids)

    seqs = tf.keras.preprocessing.sequence.pad_sequences(
            seqs, maxlen=maxlen, padding='post', value=0)

    return seqs

def filter_embeddings(embeddings, vocab, num_words, dim=300):
    """学習済み単語ベクトル表現の中から与えらえた単語ベクトルのみを抽出する"""
    _embeddings = np.random.normal(0, dim**(-0.5), size=(num_words, dim))

    for i, word in enumerate(vocab):
        # <PAD>, <UNK>, <CLS>は無視
        if i <= 2:
            continue

        if word in embeddings:
            word_id = vocab[word]
            if word_id >= num_words:
                continue
            _embeddings[word_id] = embeddings[word]

    return _embeddings.astype(np.float32)


if __name__ == '__main__':
    main()
