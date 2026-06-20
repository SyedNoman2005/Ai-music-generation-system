import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, Embedding, Attention, GlobalAveragePooling1D, Concatenate

def create_lstm_network(sequence_length, n_vocab, embedding_dim=256):
    """
    Creates the Bidirectional LSTM model with Self-Attention.
    """
    inputs = Input(shape=(sequence_length,))
    x = Embedding(input_dim=n_vocab, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    
    lstm_out1 = Bidirectional(LSTM(512, return_sequences=True))(x)
    lstm_out1 = Dropout(0.3)(lstm_out1)
    
    lstm_out2 = Bidirectional(LSTM(512, return_sequences=True))(lstm_out1)
    lstm_out2 = Dropout(0.3)(lstm_out2)
    
    attention_out = Attention()([lstm_out2, lstm_out2])
    concat_out = Concatenate()([lstm_out2, attention_out])
    
    pooled = GlobalAveragePooling1D()(concat_out)
    
    dense1 = Dense(256, activation='relu')(pooled)
    dense1 = Dropout(0.3)(dense1)
    
    outputs = Dense(n_vocab, activation='softmax')(dense1)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
