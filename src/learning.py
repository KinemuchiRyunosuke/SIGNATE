import tensorflow as tf
import tensorflow_addons as tfa

from transformer.transformer import ClassificationTransformer


def define_model(num_words, hopping_num, head_num, hidden_dim,
                 dropout_rate, learning_rate, embeddings, num_classes=2):
    model = ClassificationTransformer(
                num_classes=num_classes,
                vocab_size=num_words,
                hopping_num=hopping_num,
                head_num=head_num,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
                embeddings=embeddings)

    model.compile(optimizer=tf.keras.optimizers.Adam(
                                learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tfa.metrics.F1Score(num_classes=num_classes,
                                               average='macro')])

    return model

def train_model(model, x_train, y_train, x_val, y_val,
                batch_size, epochs, checkpoint_path):
    """Transformerモデルを学習させる"""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_score', mode='max', patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_f1_score', mode='max',
            save_best_only=True)
    ]

    model.fit(x=x_train,
              y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_val, y_val),
              callbacks=callbacks,
              shuffle=True)

    # 最も性能が良かったエポックの重みを読み込む
    model.load_weights(checkpoint_path)

    return model
