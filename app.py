from flask import Flask, render_template, request, redirect
import os
from datetime import datetime
import torch
import torchvision.transforms as transforms
from PIL import Image
from encoders.encoders import timm_backbones
from omegaconf import OmegaConf

# Flask app configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config():
    config_path = "configs/test.yaml"
    cfg = OmegaConf.load(config_path)
    return cfg


cfg = load_config()


def load_model():
    model = timm_backbones(
        encoder=cfg.model.encoder,
        num_classes=cfg.num_classes,
        optimizer_cfg=cfg.model.optimizer
    )
    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


MODEL = load_model()

CLASS_NAMES = ['Melanoma', 'Basal Cell Carcinoma']


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor


def predict_class(input_tensor):
    with torch.no_grad():
        output = MODEL(input_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(prob, dim=1)
        return predicted_idx.item(), confidence.item() * 100


def analyze_image(image_path):
    try:
        input_tensor = preprocess_image(image_path)
        predicted_idx, confidence = predict_class(input_tensor)
        predicted_class_name = CLASS_NAMES[predicted_idx]
        
        # Recommendation Logic
        if confidence >= 80:
            recommendation = ("High confidence. Please consult a specialist "
                            "for further diagnosis.")
        elif 50 <= confidence < 80:
            recommendation = ("Moderate confidence. Consider monitoring and "
                            "consulting a doctor if needed.")
        else:
            recommendation = ("Low confidence. Retake the image in better "
                            "lighting and try again.")
        
        return {
            'condition': predicted_class_name,
            'class': predicted_idx,
            'confidence': round(confidence, 2),
            'recommendation': recommendation
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None


def allowed_file(filename):
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS'])


# Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Get analysis results
        result = analyze_image(save_path)

        return render_template(
            'results.html',
            filename=filename,
            result=result,
            datetime=datetime
        )
    return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=True)