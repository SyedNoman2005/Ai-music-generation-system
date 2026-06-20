import pickle
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_processing import prepare_sequences
from model import create_network

def plot_history(history, save_dir="data/models"):
    """
    Plots the training loss and accuracy and saves the plot as an image.
    """
    plt.figure(figsize=(12, 4))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_history.png"))
    print(f"Training history plot saved to {save_dir}/training_history.png")

def train_network():
    """
    Train the advanced Neural Network to generate music
    """
    if not os.path.exists("data/models"):
        os.makedirs("data/models")
        
    try:
        with open('data/processed/features.pkl', 'rb') as filepath:
            features = pickle.load(filepath)
    except Exception as e:
        print("Could not load features.pkl. Make sure to run data_processing.py first.")
        return

    n_vocab = len(set(features))
    sequence_length = 100
    
    print(f"Loaded {len(features)} tokens. Vocabulary size: {n_vocab}")
    
    network_input, network_output = prepare_sequences(features, n_vocab, sequence_length=sequence_length)
    
    print("Building advanced model architecture (Bidirectional LSTM + Attention)...")
    model = create_network(sequence_length=sequence_length, n_vocab=n_vocab)
    model.summary()
    
    filepath = "data/models/weights-best.keras"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    callbacks_list = [checkpoint, early_stopping]
    
    print("Starting training...")
    history = model.fit(
        network_input, 
        network_output, 
        epochs=100, 
        batch_size=128, 
        callbacks=callbacks_list
    )
    
    # Plot and save history
    plot_history(history)
    
    # Save the final model
    model.save("data/models/advanced_final_model.keras")
    print("Training complete. Model saved to data/models/advanced_final_model.keras")

if __name__ == '__main__':
    train_network()
