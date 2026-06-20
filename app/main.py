from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import shutil
from visualize import plot_piano_roll
import uvicorn

app = FastAPI(title="Ultimate Music Generation API")

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory="app/static"), name="static")
if not os.path.exists("outputs"):
    os.makedirs("outputs")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

class GenerateRequest(BaseModel):
    temperature: float = 1.0
    length: int = 500
    engine: str = "lstm"

@app.get("/")
def read_root():
    return FileResponse("app/static/index.html")

def mock_generate(output_filename):
    """
    Mock generation that copies the sample MIDI file when TensorFlow is missing.
    """
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    sample_path = "data/midi_songs/sample.mid"
    output_path = os.path.join("outputs", output_filename)
    
    if os.path.exists(sample_path):
        shutil.copy(sample_path, output_path)
        return output_path
    return None

@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    output_filename = f"generated_{req.engine}_temp_{req.temperature}.mid"
    image_filename = f"piano_roll_{req.engine}_{req.temperature}.png"
    
    try:
        # Attempt to use real generator, fallback to mock if TF fails
        try:
            import tensorflow as tf
            from generate import generate_music
            model_path = "data/models/weights-best.keras"
            features_path = "data/processed/features.pkl"
            
            if not os.path.exists(model_path) or not os.path.exists(features_path):
                # If no model is trained yet, fallback to mock
                output_path = mock_generate(output_filename)
            else:
                output_path = generate_music(
                    model_path=model_path,
                    features_path=features_path,
                    temperature=req.temperature,
                    generate_length=req.length,
                    output_filename=output_filename
                )
        except ImportError:
            print("TensorFlow not found. Using Mock Generator.")
            output_path = mock_generate(output_filename)
            
        if output_path and os.path.exists(output_path):
            img_path = plot_piano_roll(output_path, f"outputs/{image_filename}")
            
            response = {
                "status": "success", 
                "file_url": f"/api/download/{output_filename}"
            }
            if img_path:
                response["image_url"] = f"/outputs/{image_filename}"
                
            return response
        else:
            raise HTTPException(status_code=500, detail="Generation failed internally. Missing sample MIDI?")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join("outputs", filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename, media_type='audio/midi')
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
