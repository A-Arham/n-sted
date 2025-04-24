# main.py
from fastapi import FastAPI, UploadFile, File
import tempfile
from inference import test_pipeline  # Import the test pipeline from inference.py

# Initialize FastAPI app
app = FastAPI()

@app.post("/run_inference/")
async def run_inference(file: UploadFile = File(...)):
    """
    Run inference on uploaded EEG .mat file.
    """
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    # Call the test pipeline from inference.py
    predictions = test_pipeline(temp_file_path)

    # Return the predictions (for example, a JSON response)
    return {"predictions": predictions}
