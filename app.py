from fastapi import FastAPI, File, UploadFile, Request,HTTPException
from io import BytesIO
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import base64

app = FastAPI()
templates = Jinja2Templates(directory="templates")  # Assuming your templates are in a directory named "templates"

# Load the trained model
model = tf.keras.models.load_model('final_model.h5')  # Update with your model path

# Define class labels
class_labels = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']

def process_image(file):
    image = Image.open(io.BytesIO(file))
    image = image.resize((256, 256))  # Assuming input size of your model
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalization
    return image


@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        # Read the image file
        image_data = await file.read()

        # Process the image
        image = process_image(image_data)

        # Make prediction
        prediction = model.predict(image)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = round(float(np.max(prediction)), 4)

        # Encode the image bytes as base64
        encoded_image = base64.b64encode(image_data).decode('utf-8')

        # Pass the image data and other details to the template
        return templates.TemplateResponse("prediction.html", {"request": request, "encoded_image": encoded_image, "predicted_class": predicted_class, "confidence": confidence})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def upload_file(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})
