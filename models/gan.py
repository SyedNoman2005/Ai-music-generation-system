import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential, Model

def build_generator(latent_dim, sequence_length, n_vocab):
    """
    Builds the Generator model for the Music GAN.
    """
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(sequence_length * n_vocab, activation='sigmoid'))
    model.add(Reshape((sequence_length, n_vocab)))
    
    noise = Input(shape=(latent_dim,))
    music_sequence = model(noise)
    return Model(noise, music_sequence)

def build_discriminator(sequence_length, n_vocab):
    """
    Builds the Discriminator model for the Music GAN.
    """
    model = Sequential()
    model.add(Flatten(input_shape=(sequence_length, n_vocab)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(1, activation='sigmoid'))
    
    music_sequence = Input(shape=(sequence_length, n_vocab))
    validity = model(music_sequence)
    return Model(music_sequence, validity)

class MusicGAN:
    def __init__(self, sequence_length, n_vocab, latent_dim=100):
        self.sequence_length = sequence_length
        self.n_vocab = n_vocab
        self.latent_dim = latent_dim
        
        self.optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        
        # Build and compile discriminator
        self.discriminator = build_discriminator(self.sequence_length, self.n_vocab)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
        # Build generator
        self.generator = build_generator(self.latent_dim, self.sequence_length, self.n_vocab)
        
        # Generator takes noise as input
        z = Input(shape=(self.latent_dim,))
        generated_music = self.generator(z)
        
        # For the combined model, train only the generator
        self.discriminator.trainable = False
        validity = self.discriminator(generated_music)
        
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)
