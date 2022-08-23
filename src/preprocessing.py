import numpy as np
import tensorflow as tf
import re

from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split


def preprocess_transformer_train(x_train, y_train, tokenizer):
    """データセットをTransformerに入力できる状態にする

    Args:
        x_train(ndarray): 文章データ
        y_train(ndarray): ラベルデータ

    Returns:
        x_train, y_train(ndarray): 学習に用いるデータセット
        x_val, y_val(ndarray): 検証用データセット
        embeddings(ndarray): 単語分散表現
        tokenizer: tf.keras.preprocessing.text.tokenizerのインスタンス．
            x_trainでfittingを行ったもの．

    """
    # MLを陽性，それ以外を陰性にする
    y_train = (y_train == 2).astype(int)

    # htmlタグを削除
    x_train = clean_html(x_train)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=1)

    # 各単語をIDに変換
    tokenizer.fit_on_texts(x_train)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_val = tokenizer.texts_to_sequences(x_val)

    # Padding
    maxlen = max(list(map(lambda seq: len(seq), x_train)))
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
            x_train, maxlen=maxlen+10, padding='post', value=0)
    x_val = tf.keras.preprocessing.sequence.pad_sequences(
            x_val, maxlen=maxlen+10, padding='post', value=0)

    # One-Hot Encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=2)

    return x_train, y_train, x_val, y_val, tokenizer, maxlen

def preprocess_test_transformer(x_test, tokenizer, maxlen):
    """テストデータをTransformerに入力できるようにする

    Args:
        x_test(ndarray): テストデータ
        tokenizer: 学習用データに対してfittingを行った
            tf.keras.preprocessing.text.tokenizerのインスタンス．
        maxlen(int): paddingを行う長さ

    """
    # htmlタグを削除
    x_test = clean_html(x_test)

    x_test = tokenizer.texts_to_sequences(x_test)

    # Padding
    x_test = tf.keras.preprocessing.sequence.pad_sequences(
            x_test, maxlen=maxlen+10, padding='post', value=0)

    return x_test

def preprocess_train_svm(x_train, stemmer, tf_vect):
    x_train = clean(x_train, stemmer)
    x_train = tf_vect.fit_transform(x_train)
    return x_train, tf_vect

def preprocess_test_svm(x_test, stemmer, tf_vect):
    x_test = clean(x_test, stemmer)
    x_test = tf_vect.transform(x_test)
    return x_test

def clean_html(x):
    for i, text in enumerate(x):
        soup = BeautifulSoup(text, 'html.parser')
        x[i] = soup.get_text()

    return x

def convert_examples_to_features(x, maxlen, tokenizer):
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

def clean(texts, stemmer):
    clean_texts = []
    for text in texts:
        # htmlタグを削除
        text = remove_tag(text)
        #アルファベット以外をスペースに置き換え
        clean_punc = re.sub(r'[^a-zA-Z]', ' ', text)
        #単語長が3文字以下のものは削除する
        clean_short_tokenized = [word for word in clean_punc.split() if len(word) > 3]
        #ステミング
        clean_normalize = [stemmer.stem(word) for word in clean_short_tokenized]
        #単語同士をスペースでつなぎ, 文章に戻す
        clean_text = ' '.join(clean_normalize)
        clean_texts.append(clean_text)
    return clean_texts

def remove_tag(x):
    p = re.compile(r"<[^>]*?>")
    return p.sub('',x)

