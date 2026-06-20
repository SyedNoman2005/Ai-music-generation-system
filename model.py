import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, Embedding, Attention, GlobalAveragePooling1D, Concatenate

def create_network(sequence_length, n_vocab, embedding_dim=256):
    """
    Creates an advanced neural network architecture with Bidirectional LSTMs and Attention.
    Uses the Functional API.
    """
    # Input layer for integer sequences
    inputs = Input(shape=(sequence_length,))
    
    # Embedding layer to learn dense vector representations of the tokens
    x = Embedding(input_dim=n_vocab, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    
    # First Bidirectional LSTM layer
    # return_sequences=True is required to feed into the next LSTM or Attention layer
    lstm_out1 = Bidirectional(LSTM(512, return_sequences=True))(x)
    lstm_out1 = Dropout(0.3)(lstm_out1)
    
    # Second Bidirectional LSTM layer
    lstm_out2 = Bidirectional(LSTM(512, return_sequences=True))(lstm_out1)
    lstm_out2 = Dropout(0.3)(lstm_out2)
    
    # Self-Attention Mechanism
    # The Attention layer takes a list of inputs: [query, value]
    # For self-attention, query and value are the same.
    attention_out = Attention()([lstm_out2, lstm_out2])
    
    # We can concatenate the attention output with the original LSTM output for a residual-like connection
    concat_out = Concatenate()([lstm_out2, attention_out])
    
    # Pool the sequence down to a single vector per batch
    pooled = GlobalAveragePooling1D()(concat_out)
    
    # Dense layers
    dense1 = Dense(256, activation='relu')(pooled)
    dense1 = Dropout(0.3)(dense1)
    
    # Output layer
    outputs = Dense(n_vocab, activation='softmax')(dense1)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Test model shape
    model = create_network(100, 500)
    model.summary()
