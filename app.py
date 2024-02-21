import torch
from PIL import Image
from torchvision import models, transforms
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load the pre-trained model and set it to evaluation mode
model = models.resnet50(pretrained=True)
model.eval()

# Define the transformations to apply to the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict the image class and return the result
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities[predicted_class].item()

@app.route('/')
def upload_file():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file_redirect():
   if request.method == 'POST':
      f = request.files['file']
      filename = secure_filename(f.filename)
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      return redirect(url_for('results', filename=filename))

@app.route('/results/<filename>')
def results(filename):
    predicted_class, probability = predict_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('results.html', filename=filename, class_label=predicted_class, probability=probability)

if __name__ == '__main__':
   app.run(debug = True)
