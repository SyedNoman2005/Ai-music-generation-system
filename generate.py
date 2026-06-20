import pickle
import numpy as np
from music21 import instrument, note, stream, chord, duration
import os
from model import create_network

def prepare_sequences(features, feature_names, n_vocab, sequence_length=100):
    """
    Prepare the sequences used by the Neural Network
    """
    feature_to_int = dict((feature, number) for number, feature in enumerate(feature_names))
    
    network_input = []
    for i in range(0, len(features) - sequence_length, 1):
        sequence_in = features[i:i + sequence_length]
        network_input.append([feature_to_int[char] for char in sequence_in])
        
    return network_input

def sample(preds, temperature=1.0):
    """
    Helper function to sample an index from a probability array, using temperature.
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_features(model, network_input, feature_names, n_vocab, generate_length=500, temperature=1.0):
    """
    Generate Pitch_Duration tokens from the neural network based on a sequence
    """
    # Pick a random sequence from the input as a starting point
    start = np.random.randint(0, len(network_input)-1)
    
    int_to_feature = dict((number, feature) for number, feature in enumerate(feature_names))
    
    pattern = network_input[start]
    prediction_output = []
    
    # Generate features
    for note_index in range(generate_length):
        # Shape needs to be (1, sequence_length) for Embedding layer
        prediction_input = np.reshape(pattern, (1, len(pattern)))
        
        prediction = model.predict(prediction_input, verbose=0)[0]
        
        # Apply temperature sampling
        index = sample(prediction, temperature)
        
        result = int_to_feature[index]
        prediction_output.append(result)
        
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        
    return prediction_output

def parse_fraction(s):
    try:
        if '/' in s:
            num, den = s.split('/')
            return float(num) / float(den)
        return float(s)
    except:
        return 0.5 # default duration

def create_midi(prediction_output, output_filename='output.mid'):
    """
    Convert the Pitch_Duration tokens to notes and create a midi file
    """
    offset = 0
    output_notes = []
    
    for token in prediction_output:
        # Parse token, e.g., "C4_1.0" or "4.7.11_0.5" or "C4_1/3"
        parts = token.split('_')
        pitch_info = parts[0]
        dur_info = parts[1] if len(parts) > 1 else "0.5"
        
        dur_value = parse_fraction(dur_info)
        
        # If it's a chord
        if ('.' in pitch_info) or pitch_info.isdigit():
            notes_in_chord = pitch_info.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.duration = duration.Duration(dur_value)
            output_notes.append(new_chord)
        # If it's a note
        else:
            try:
                new_note = note.Note(pitch_info)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                new_note.duration = duration.Duration(dur_value)
                output_notes.append(new_note)
            except:
                pass # skip invalid notes
            
        # Increase offset based on the duration of the current note
        offset += dur_value
        
    midi_stream = stream.Stream(output_notes)
    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    output_path = os.path.join("outputs", output_filename)
    midi_stream.write('midi', fp=output_path)
    return output_path

def generate_music(model_path="data/models/weights-best.keras", 
                   features_path="data/processed/features.pkl", 
                   temperature=1.0, 
                   generate_length=500,
                   output_filename='generated_music.mid'):
    """
    Main entry point for music generation
    """
    print("Loading data and model...")
    try:
        with open(features_path, 'rb') as filepath:
            features = pickle.load(filepath)
    except Exception as e:
        print(f"Error loading {features_path}: {e}")
        return None
        
    feature_names = sorted(set(item for item in features))
    n_vocab = len(set(features))
    sequence_length = 100
    
    network_input = prepare_sequences(features, feature_names, n_vocab, sequence_length)
    
    # Recreate model architecture
    model = create_network(sequence_length=sequence_length, n_vocab=n_vocab)
    try:
        model.load_weights(model_path)
    except Exception as e:
        print(f"Error loading model weights from {model_path}: {e}")
        return None
        
    print(f"Generating sequence of length {generate_length} with temperature {temperature}...")
    prediction_output = generate_features(model, network_input, feature_names, n_vocab, generate_length, temperature)
    
    print("Converting sequence to MIDI file...")
    output_path = create_midi(prediction_output, output_filename)
    print(f"Music generated successfully and saved to {output_path}")
    return output_path

if __name__ == '__main__':
    generate_music()
