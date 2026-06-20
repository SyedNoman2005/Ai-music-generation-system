import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
import os

def get_notes(midi_path="data/midi_songs"):
    """
    Extracts notes and chords from all midi files in the specified directory.
    """
    notes = []
    
    if not os.path.exists(midi_path):
        os.makedirs(midi_path)
        print(f"Created directory {midi_path}. Please add MIDI files there.")
        return notes

    files = glob.glob(os.path.join(midi_path, "*.mid"))
    if not files:
        print(f"No MIDI files found in {midi_path}. Please add some!")
        return notes
        
    for file in files:
        print(f"Parsing {file}")
        midi = converter.parse(file)
        
        notes_to_parse = None
        
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
            
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
        
    with open('data/processed/notes.pkl', 'wb') as filepath:
        pickle.dump(notes, filepath)
        
    return notes

def prepare_sequences(notes, n_vocab, sequence_length=100):
    """
    Prepare the sequences used by the Neural Network.
    """
    # get all pitch names
    pitchnames = sorted(set(item for item in notes))
    
    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    network_input = []
    network_output = []
    
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
        
    n_patterns = len(network_input)
    
    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # normalize input
    network_input = network_input / float(n_vocab)
    
    # one-hot encode the output
    import tensorflow as tf
    network_output = tf.keras.utils.to_categorical(network_output, num_classes=n_vocab)
    
    return (network_input, network_output)

if __name__ == '__main__':
    notes = get_notes()
    if notes:
        n_vocab = len(set(notes))
        print(f"Extracted {len(notes)} total notes. Unique notes (vocabulary): {n_vocab}")
        network_input, network_output = prepare_sequences(notes, n_vocab)
        print(f"Sequence preparation complete. Input shape: {network_input.shape}")
