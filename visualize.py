import matplotlib.pyplot as plt
import pretty_midi
import numpy as np

def plot_piano_roll(midi_file_path, output_image_path="piano_roll.png"):
    """
    Reads a MIDI file and plots its piano-roll representation.
    Returns the path to the saved image file.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        
        # We'll plot the piano roll of the first instrument
        if len(midi_data.instruments) == 0:
            return None
            
        instrument = midi_data.instruments[0]
        piano_roll = instrument.get_piano_roll(fs=100)
        
        # Replace 0s with NaNs so they show as white
        piano_roll[piano_roll == 0] = np.nan
        
        plt.figure(figsize=(12, 6))
        plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
        
        plt.xlabel("Time (frames)")
        plt.ylabel("Pitch (MIDI Note Number)")
        plt.title("Piano-Roll Visualization")
        plt.colorbar(label='Velocity')
        
        plt.tight_layout()
        plt.savefig(output_image_path)
        plt.close()
        
        return output_image_path
    except Exception as e:
        print(f"Failed to create piano roll: {e}")
        return None
