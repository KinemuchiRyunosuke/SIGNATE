import tensorflow as tf
import math

PAD_ID = 0


class TokenEmbedding(tf.keras.layers.Layer):
    '''
    トークン列を Embedded Vector 列に変換します。
    '''
    def __init__(self, vocab_size: int, embedding_dim: int,
                 dtype=tf.float32, embeddings=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.dtype_ = dtype
        self.embeddings = embeddings

    def build(self, input_shape: tf.TensorShape) -> None:
        if self.embeddings is None:
            self.lookup_table = self.add_weight(
                name='token_embedding',
                shape=[self.vocab_size, self.embedding_dim],
                dtype=self.dtype_,
                initializer=tf.random_normal_initializer(
                                0., self.embedding_dim ** -0.5),
            )
        else:
            self.lookup_table = self.set_weights(self.embeddings)

        super().build(input_shape)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        mask = tf.compat.v1.to_float(tf.not_equal(input, PAD_ID))
        embedding = tf.compat.v1.nn.embedding_lookup(
                        self.lookup_table, input)
        # 元々 PAD だった部分を0にする
        embedding *= tf.expand_dims(mask, -1)
        return embedding * self.embedding_dim ** 0.5


class AddPositionalEncoding(tf.keras.layers.Layer):
    '''
    入力テンソルに対し、位置の情報を付与して返すレイヤーです。
    see: https://arxiv.org/pdf/1706.03762.pdf

    PE_{pos, 2i}   = sin(pos / 10000^{2i / d_model})
    PE_{pos, 2i+1} = cos(pos / 10000^{2i / d_model})
    '''
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        fl_type = inputs.dtype
        batch_size, max_length, depth = tf.unstack(tf.shape(inputs))

        depth_counter = tf.range(depth) // 2 * 2  # 0, 0, 2, 2, 4, ...
        depth_matrix = tf.tile(tf.expand_dims(depth_counter, 0),
                    [max_length, 1])  # [max_length, depth]
        depth_matrix = tf.pow(10000.0, tf.cast(
                    depth_matrix / depth, fl_type)) # [max_length, depth]

        # cos(x) == sin(x + π/2)
        # 0, π/2, 0, π/2, ...
        phase = tf.cast(tf.range(depth) % 2, fl_type) * math.pi / 2
        # [max_length, depth]
        phase_matrix = tf.tile(tf.expand_dims(phase, 0), [max_length, 1])

        pos_counter = tf.range(max_length)
        pos_matrix = tf.cast(tf.tile(tf.expand_dims(pos_counter, 1),
                        [1, depth]), fl_type)  # [max_length, depth]

        positional_encoding = tf.sin(pos_matrix / depth_matrix + phase_matrix)
        # [batch_size, max_length, depth]
        positional_encoding = tf.tile(tf.expand_dims(positional_encoding, 0),
                                      [batch_size, 1, 1])

        return inputs + positional_encoding
