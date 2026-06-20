import urllib.request
import os

def download_sample_midi():
    """
    Downloads a sample MIDI file to the data/midi_songs directory.
    """
    if not os.path.exists("data/midi_songs"):
        os.makedirs("data/midi_songs")
        
    url = "https://bitmidi.com/uploads/15105.mid" # Example: Super Mario Bros theme
    output_path = "data/midi_songs/sample.mid"
    
    print(f"Downloading sample MIDI from {url}...")
    try:
        # Use a user-agent to avoid 403 Forbidden
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Sample MIDI downloaded to {output_path}")
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    download_sample_midi()
