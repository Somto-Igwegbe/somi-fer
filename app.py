from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import subprocess
import requests
import time
import io
from PIL import Image
import base64

ngrok_process = subprocess.Popen(["C:\ProgramData\chocolatey\\bin\\ngrok.exe", 'http', '8080'])
time.sleep(2)
ngrok_url = requests.get('http://localhost:4040/api/tunnels').json()['tunnels'][0]['public_url']
print('Ngrok URL:', ngrok_url)

import torch
import torch.nn as nn
from torchvision.transforms import transforms

class SomiCNN(nn.Module):
    def __init__(self, num_classes):
        super(SomiCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 37 * 37, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x

num_classes = 7
somi_model = SomiCNN(num_classes=num_classes)

state_dict = torch.load("C:\\Users\\ogeum\\OneDrive\\Documents\\somi_bestmodel_adam.pth", map_location=torch.device('cpu'))
somi_model.load_state_dict(state_dict['model_state_dict'])
somi_model.eval()

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

class_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = somi_model(image)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def post_endpoint():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        try:
            img_bytes = file.read()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            image = Image.open(io.BytesIO(img_bytes))
            predicted_label = class_labels[predict(image)]
            return render_template('index.html', predicted_label=predicted_label, image=img_base64)
        except Exception as e:
            return render_template('index.html', error='Error processing image')
    return render_template('index.html', predicted_label=None, image=None)



if __name__ == '__main__':
    app.run(port=8080)
