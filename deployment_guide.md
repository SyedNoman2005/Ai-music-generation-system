# Deployment Guide: AI Symphony (Music Generation)

This guide covers deploying the FastAPI + HTML/JS application to the cloud so you can show it off in your portfolio.

## Option 1: Hugging Face Spaces (Recommended for AI Portfolios)

Hugging Face Spaces is perfect because it's designed for AI models, provides a free tier, and looks great on a resume.

1. **Create a Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space.
   - Choose **Docker** as the SDK.

2. **Add a `Dockerfile`** to the root of your project:
   ```dockerfile
   FROM python:3.10
   
   WORKDIR /code
   
   COPY ./requirements.txt /code/requirements.txt
   RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
   
   COPY . /code
   
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
   ```

3. **Upload Files**:
   Upload all your project files (including `data/models/weights-best.keras` and `data/processed/features.pkl`) to the Space repository. Hugging Face will automatically build the Docker image and deploy the FastAPI server.

## Option 2: Render.com (Great for general Web Apps)

1. Create a new "Web Service" on Render and link your GitHub repository containing this code.
2. Setup the configuration:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3. Click deploy!

> **Note on Model Hosting**:
> Deep learning models (`.keras` files) can be quite large. If your model exceeds GitHub's file size limits (100MB), you will need to use Git LFS (Large File Storage) or download the model from an external source (like AWS S3 or Hugging Face Models) inside your startup script.
