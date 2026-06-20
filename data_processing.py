import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, duration
import os

def extract_features(midi_path="data/midi_songs"):
    """
    Extracts notes, chords, and durations from all midi files in the specified directory.
    Combines them into a string format: 'Pitch_Duration'
    """
    features = []
    
    if not os.path.exists(midi_path):
        os.makedirs(midi_path)
        print(f"Created directory {midi_path}. Please add MIDI files there.")
        return features

    files = glob.glob(os.path.join(midi_path, "*.mid"))
    if not files:
        print(f"No MIDI files found in {midi_path}. Please add some!")
        return features
        
    for file in files:
        print(f"Parsing {file}")
        try:
            midi = converter.parse(file)
            notes_to_parse = None
            
            try: # file has instrument parts
                s2 = instrument.partitionByInstrument(midi)
                # Attempt to find piano parts first
                piano_parts = [p for p in s2.parts if 'Piano' in str(p)]
                if piano_parts:
                    notes_to_parse = piano_parts[0].recurse()
                else:
                    notes_to_parse = s2.parts[0].recurse() 
            except: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
                
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    # Combine pitch and duration
                    token = f"{str(element.pitch)}_{element.duration.quarterLength}"
                    features.append(token)
                elif isinstance(element, chord.Chord):
                    # Combine chord normal order and duration
                    token = f"{'.'.join(str(n) for n in element.normalOrder)}_{element.duration.quarterLength}"
                    features.append(token)
        except Exception as e:
            print(f"Error parsing {file}: {e}")
                
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
        
    with open('data/processed/features.pkl', 'wb') as filepath:
        pickle.dump(features, filepath)
        
    return features

def prepare_sequences(features, n_vocab, sequence_length=100):
    """
    Prepare the sequences used by the Neural Network.
    Since we use Embeddings now, we don't normalize the input to [0,1], 
    we just return the raw integer sequences for the Embedding layer.
    """
    # get all feature names
    feature_names = sorted(set(item for item in features))
    
    # create a dictionary to map features to integers
    feature_to_int = dict((feature, number) for number, feature in enumerate(feature_names))
    
    network_input = []
    network_output = []
    
    # create input sequences and the corresponding outputs
    for i in range(0, len(features) - sequence_length, 1):
        sequence_in = features[i:i + sequence_length]
        sequence_out = features[i + sequence_length]
        network_input.append([feature_to_int[char] for char in sequence_in])
        network_output.append(feature_to_int[sequence_out])
        
    n_patterns = len(network_input)
    
    # reshape the input into a format compatible with Embedding layers (n_patterns, sequence_length)
    network_input = np.reshape(network_input, (n_patterns, sequence_length))
    
    # one-hot encode the output
    import tensorflow as tf
    network_output = tf.keras.utils.to_categorical(network_output, num_classes=n_vocab)
    
    return (network_input, network_output)

if __name__ == '__main__':
    features = extract_features()
    if features:
        n_vocab = len(set(features))
        print(f"Extracted {len(features)} total tokens. Unique tokens (vocabulary): {n_vocab}")
        network_input, network_output = prepare_sequences(features, n_vocab)
        print(f"Sequence preparation complete. Input shape: {network_input.shape}")
