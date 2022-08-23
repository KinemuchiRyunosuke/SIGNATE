import os
import pandas as pd
import gensim
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

from preprocessing import *
from learning import *

num_classes = 4

# parameter for transformer
batch_size = 16
epochs = 100
num_words = 6000
hopping_num = 4
head_num = 10
hidden_dim = 300
dropout_rate = 0.1
learning_rate = 1.0e-5

# parameter for SVM
C = 20
gamma = 0.5
kernel = 'rbf'
probability = True

train_dataset_path = "data/train.csv"
test_dataset_path = "data/test.csv"
pretrained_model_name_or_path = 'bert-base-uncased'
word_vectors_path = "data/wiki-news-300d-1M.vec"
checkpoint_path = "models/checkpoint"
submit_path = "data/submit.csv"


def main():
    # csvデータの読み込み
    df_train = pd.read_csv(train_dataset_path)
    df_test = pd.read_csv(test_dataset_path)

    # 提出用データフレーム
    jobflags = np.zeros(df_test.shape[0], dtype=int)
    df_submit = pd.DataFrame([df_test['id'], jobflags],
                             index=['id', 'jobflag']).T

    ##################################################################
    # Transformer で ML(機械学習エンジニア)を判別                      #
    ##################################################################
    x_train = df_train['description'].to_numpy()
    y_train = df_train['jobflag'].to_numpy()
    x_test = df_test['description'].to_numpy()

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=num_words,
        lower=True,
        oov_token='<UNK>'
    )

    # 各種トークンのIDを設定
    vocab = {'<PAD>':0, '<UNK>':1, '<CLS>':2}
    tokenizer.word_index = vocab

    # データセットをTransformerに入力できる状態にする
    x_train, y_train, x_val, y_val, tokenizer, maxlen = \
            preprocess_transformer_train(x_train, y_train, tokenizer)
    x_test = preprocess_test_transformer(x_test, tokenizer, maxlen)

    if os.path.exists(checkpoint_path):
        # Transformerモデルを定義
        model = define_model(num_words, hopping_num, head_num, hidden_dim,
                             dropout_rate, learning_rate, embeddings=None)

        model.load_weights(checkpoint_path)

    else:
        # 学習済み単語分散表現を読み込む
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
                        word_vectors_path)
        word_vectors = filter_embeddings(word_vectors,
                                         tokenizer.word_index,
                                         num_words)

        # Transformerモデルを定義
        model = define_model(num_words, hopping_num, head_num, hidden_dim,
                             dropout_rate, learning_rate,
                             embeddings=word_vectors)

        # Transformerを学習
        model = train_model(model, x_train, y_train, x_val, y_val,
                            batch_size, epochs, checkpoint_path)

    # TransformerでMLを判別
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    # MLと判別されたデータを2でラベルづけ
    df_submit.loc[(y_pred == 1), 'jobflag'] = 2

    ##################################################################
    # SVM で3値分類                                                   #
    ##################################################################
    # MLでないと判別されたデータだけ抽出
    df_test = df_test.loc[(y_pred == 0), :]
    x_test = df_test['description'].to_numpy()

    # SVM学習用データ
    x_train = df_train['description'].to_numpy()
    y_train = df_train['jobflag'].to_numpy()
    x_train = x_train[y_train != 2]     # ML(機械学習エンジニア）でないもののみを
    y_train = y_train[y_train != 2]     # SVMの学習に用いる

    # ラベルを [1, 3, 4] から [0, 1, 2] に変換
    label_dict = {1:0, 3:1, 4:2}
    y_train = np.array(list(map(lambda y: label_dict[y], y_train)))

    # SVMに入力できるようにデータを加工
    stemmer = PorterStemmer()
    tf_vect = TfidfVectorizer()
    x_train, tf_vect = preprocess_train_svm(x_train, stemmer, tf_vect)
    x_test = preprocess_test_svm(x_test, stemmer, tf_vect)

    # SVMを学習
    model_svm = svm.SVC(C=C, gamma=gamma, kernel=kernel, probability=probability)
    model_svm.fit(x_train, y_train)
    y_pred = model_svm.predict(x_test)

    # ラベルを [0, 1, 2] から [1, 3, 4] に変換
    y_pred = np.array(list(map(
        lambda y: inverse_lookup(label_dict, y), y_pred)))

    # SVMの判別結果をデータフレームに書き込む
    indices = df_test.index
    df_submit.loc[indices, 'jobflag'] = y_pred

    # データフレームをCSVにして保存
    df_submit.to_csv(submit_path, header=None, index=None)


def inverse_lookup(dict, x):
    """辞書を逆引き"""
    for key, val in dict.items():
        if x == val:
            return key

if __name__ == '__main__':
    main()
