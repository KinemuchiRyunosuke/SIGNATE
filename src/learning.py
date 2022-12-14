import pickle
import tensorflow as tf
import tensorflow_addons as tfa

from transformer.transformer import ClassificationTransformer

num_classes = 4
batch_size = 16
epochs = 100
num_words = 6000
hopping_num = 4
head_num = 10
hidden_dim = 300
dropout_rate = 0.1
learning_rate = 1.0e-5

dataset_path = "data/processed/processed_dataset.pickle"
embedding_path = "data/embedding.pickle"

def main():
    # データセット読み込み
    with open(dataset_path, 'rb') as f:
        x = pickle.load(f)
        y = pickle.load(f)

    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)

    model = ClassificationTransformer(
                num_classes=num_classes,
                vocab_size=num_words,
                hopping_num=hopping_num,
                head_num=head_num,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
                embeddings=embeddings
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(
                                learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tfa.metrics.F1Score(num_classes=num_classes,
                                               average='micro')])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_score', mode='max', patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/checkpoint",
            monitor='val_f1_score', mode='max',
            save_best_only=True)
    ]

    model.fit(x=x,
              y=y,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=callbacks,
              shuffle=True)


if __name__ == '__main__':
    main()