import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Add

def transformer_decoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    A single block of a Transformer Decoder.
    """
    # Self-attention mechanism
    # Setting use_causal_mask=True ensures the model only attends to past tokens
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs, use_causal_mask=True)
    
    attention_output = Dropout(dropout)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(Add()([inputs, attention_output]))

    # Feed Forward network
    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    
    return LayerNormalization(epsilon=1e-6)(Add()([out1, ffn_output]))

def create_transformer_network(sequence_length, n_vocab, embedding_dim=256, num_heads=4, ff_dim=512, num_blocks=4):
    """
    Creates a simplified Causal Music Transformer model.
    """
    inputs = Input(shape=(sequence_length,))
    
    # Embedding layer
    x = Embedding(input_dim=n_vocab, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    
    # Adding absolute positional encoding
    positions = tf.range(start=0, limit=sequence_length, delta=1)
    position_embedding = Embedding(input_dim=sequence_length, output_dim=embedding_dim)(positions)
    x = x + position_embedding
    
    # Transformer Blocks
    for _ in range(num_blocks):
        x = transformer_decoder_block(x, head_size=embedding_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=0.1)

    # Pooling
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.1)(x)
    
    outputs = Dense(n_vocab, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = create_transformer_network(100, 500)
    model.summary()
