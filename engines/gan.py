import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, LeakyReLU, Reshape

def create_generator(latent_dim, sequence_length, n_vocab):
    """
    Creates the GAN Generator. Takes random noise and outputs a sequence of probabilities.
    """
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((1, 128)))
    
    # LSTM to expand into sequence length
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    
    # We need to output `sequence_length` steps
    # We can use a Dense layer wrapped in a TimeDistributed or just rely on LSTM returning sequences
    # Here we will generate a single sequence of `n_vocab` features step by step.
    # For a simplified discrete GAN, we output softmax probabilities over vocab.
    # Note: Training discrete GANs is extremely hard, this is a simplified baseline.
    model.add(LSTM(512, return_sequences=True))
    model.add(Dense(n_vocab, activation='softmax'))
    
    # The output is (1, n_vocab), we actually need to generate a full sequence length.
    # We will reshape and build a proper architecture for (sequence_length, n_vocab)
    
    # Redefine properly:
    inputs = Input(shape=(latent_dim,))
    x = Dense(sequence_length * 64)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((sequence_length, 64))(x)
    
    x = LSTM(256, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(256, return_sequences=True)(x)
    
    # Output layer gives a probability distribution over the vocabulary for each time step
    outputs = Dense(n_vocab, activation='softmax')(x)
    
    generator = Model(inputs, outputs)
    return generator

def create_discriminator(sequence_length, n_vocab):
    """
    Creates the GAN Discriminator. Evaluates if a sequence is real or fake.
    """
    # Discriminator receives a sequence of one-hot encoded or softmax distributions
    inputs = Input(shape=(sequence_length, n_vocab))
    
    x = LSTM(256, return_sequences=True)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    discriminator = Model(inputs, outputs)
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    
    return discriminator

def create_gan(generator, discriminator):
    """
    Combines Generator and Discriminator into a full GAN model.
    """
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
    return model

if __name__ == "__main__":
    latent_dim = 100
    seq_len = 100
    vocab = 500
    
    gen = create_generator(latent_dim, seq_len, vocab)
    gen.summary()
    
    disc = create_discriminator(seq_len, vocab)
    disc.summary()
