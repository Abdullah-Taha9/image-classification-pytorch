# Skin Cancer Classification

## Description
G4 Pro is a deep learning project for classifying Melanoma and Basal-cell carcinoma using a subset of the HAM10000 dataset. It compares ResNet50, MobileNetV3-Large-100, and EfficientNet-B0, with the best performer (MobileNetV3) deployed via a Flask web app for real-time skin lesion analysis.

---

## Prerequisites
Before using this repo, ensure you have:
- **Python 3.8+**
- **Git** installed
- **GPU** (optional, for training; CUDA required if used)
- Dependencies: `pip install -r requirements.txt` (see [Installation](#installation))

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://gitos.rrze.fau.de/utn-machine-intelligence/teaching/ml-ws2425-final-projects/g4.git
   cd g4
   ```

2. **Set Up a Virtual Environment** (recommended)
   ```bash
   conda create --name g4 python==3.9
   conda activate g4  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note*: Includes `torch`, `pytorch-lightning`, `timm`, `flask`, `kagglehub`, etc. Ensure `requirements.txt` is in the repo root.

---

## Usage

### Running the Flask App
1. **Ensure Pretrained Model is Available**
   - The default model (`model.pth`, MobileNetV3) should be in the repo root. If missing, train it first (see [Training](#training)).

2. **Start the App**
   ```bash
   python app.py
   ```
   - Opens at `http://localhost:5000`.

3. **Analyze Skin Lesions**
   - Visit the URL in a browser.
   - Upload a skin lesion image (PNG/JPG).
   - View results: classification (Melanoma/BCC), confidence, and recommendation.

### Training the Model
1. **Prepare Configuration**
   - Edit `configs/train.yaml` if needed (e.g., change `encoder` to `mobilenetv3_large_100`).

2. **Run Training**
   ```bash
   python train.py max_epochs=100 min_epochs=40 batch_size=16 model.encoder=mobilenetv3_large_100 dataset_name=HAM10000 model.optimizer.name=Adam model.optimizer.lr=3e-5 dataset.target_size=[224,224] seed=42
   ```
   - Outputs: trained model (`model.pth`), logs in `logs/outputloggs`, and a PDF report in the output directory.

3. **Expected Output**
   - Terminal shows training metrics (e.g., `Training Loss: 0.0191`, `Test Acc: 0.96`).
   - Terminal also shows validation and testing metrics.
   - Model saved to `output_dir/model.pth`.
   - pdf report saved to `output_dir/training_report.pdf`.

---

## Project Structure
- `app.py`: Flask web app for deployment.
- `train.py`: Training script using PyTorch Lightning.
- `timm_backbones.py`: Model architecture definition.
- `custom_dataset.py`: Dataset loading and preprocessing.
- `configs/`: Configuration files (e.g., `train.yaml`).
- `static/`: Web app assets (CSS, uploads).
- `templates/`: Web app html templates.
- `utils/`: Helper functions to generate the report and produce sample perdictions.

---

## Support
For issues or questions:
- Open an [issue](https://gitos.rrze.fau.de/utn-machine-intelligence/teaching/ml-ws2425-final-projects/g4/-/issues) on GitLab.
- Contact: abdullah.al-labani@utn.de, amr.reyad@utn.de, saeed.al-maqtari@utn.de

---

## Authors
- Abdullah Al-Labani
- Amr Reyad
- Saeed Al-Maqtari

## License
This project is licensed under the MIT License (pending final decision).

## Project Status
Active development as of February 10, 2025. Contributions welcome!

---
## References 
-- Used DeepSeek to help with Readme file styling and Front-end code styling.
