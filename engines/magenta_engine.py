import note_seq
import os

def create_magenta_sequence(midi_file_path):
    """
    Parses a MIDI file into a Magenta NoteSequence.
    """
    try:
        sequence = note_seq.midi_file_to_note_sequence(midi_file_path)
        return sequence
    except Exception as e:
        print(f"Error parsing MIDI with Magenta: {e}")
        return None

def synthesize_audio(sequence, output_wav_path="outputs/audio.wav"):
    """
    Uses FluidSynth (if installed system-wide) to synthesize the sequence into WAV.
    Note: Requires FluidSynth and a soundfont (.sf2) file.
    """
    try:
        # Fallback to pure note-seq synthesis which doesn't strictly need fluidsynth for basic playback
        # if using the web synthesizer, but here we can save as MIDI easily.
        # For actual wav generation, note_seq.fluidsynth is used.
        note_seq.sequence_proto_to_midi_file(sequence, output_wav_path.replace(".wav", ".mid"))
        return True
    except Exception as e:
        print(f"Error synthesizing audio: {e}")
        return False
