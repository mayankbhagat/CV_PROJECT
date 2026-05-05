# Cliniscan
# 🩺 Cliniscan – AI-Powered Chest X-ray Analysis System

CliniScan is an end-to-end deep learning system for **automated chest X-ray analysis**, combining:

- 🧠 Multi-label disease classification  
- 🎯 Lesion detection (YOLO)  
- 🔥 Explainability using Grad-CAM  
- 🧾 Automated medical report generation  
- 🏥 DICOM image support  

👉 **Live Demo:**  
🔗 https://huggingface.co/spaces/mayankbhagat/Cliniscan  

---

## 🚀 Features

- ✔ Detects multiple thoracic conditions:
  - Opacity  
  - Consolidation  
  - Fibrosis  
  - Mass  
  - Other  

- ✔ Localizes abnormalities with bounding boxes (YOLO)

- ✔ Grad-CAM heatmaps for explainability

- ✔ Upload formats:
  - `.png`, `.jpg`
  - `.dcm` (DICOM medical images)

- ✔ Auto-generated AI medical report

- ✔ Interactive UI (Gradio)

---

## 🧠 Model Architecture

| Module | Model |
|------|------|
| Classification | EfficientNet-B0 (Torchvision) |
| Detection | YOLO (Ultralytics) |
| Explainability | Grad-CAM |
| UI | Gradio |

---

## 📁 Project Structure

```bash
CliniScan/
│── app.py
│── requirements.txt
│── efficientnet_best.pt
│── yolo_best.pt
│
├── src/
│   ├── train_classification.py
│   ├── train_yolo.py
│   ├── gradcam_example.py
│   ├── run_full_pipeline.py
│
├── data/
├── plots/
├── runs/
```

---

## ⚙️ Installation (Local Setup)

### 1. Clone the repository
```bash
git clone https://github.com/your-username/CliniScan.git
cd CliniScan
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add model weights
Place these files in root directory:

```
efficientnet_best.pt
yolo_best.pt
```

### 5. Run the app
```bash
python app.py
```

Open:
```
http://localhost:7860
```

---

## 🖼️ How to Use

1. Upload a chest X-ray image  
2. Adjust Grad-CAM opacity (optional)  
3. Click **Run Analysis**  
4. View:
   - 📊 Classification probabilities  
   - 🎯 YOLO detections  
   - 🔥 Grad-CAM heatmap  
   - 🧾 Generated report  

---

## 📊 Performance

### Classification Model
- F1 Score: **~0.81**
- Sensitivity: **~0.86**
- AUC-ROC: **~0.89**

### Detection Model
- mAP@0.5: **~0.37**
- Precision: **~0.59**
- Recall: **~0.33**

---

## 🏥 DICOM Support

- VOI LUT normalization  
- Handles MONOCHROME1 / MONOCHROME2  
- Converts to RGB for model inference  

---

## ☁️ Deployment

This project is deployed using:
- Gradio  
- Hugging Face Spaces (Free Hosting)

### To deploy yourself:
1. Create Hugging Face Space (Gradio)
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `.pt` files
3. Click Deploy 🚀

---

## ⚠️ Disclaimer

> This project is for **educational and research purposes only**.  
> It is **not a substitute for professional medical diagnosis**.

---

## 🔮 Future Improvements

- PDF report export  
- Better Grad-CAM (class-specific)  
- PACS-style DICOM viewer  
- Model calibration  
- Clinical validation  
- Faster GPU deployment  

---

## 👨‍💻 Author

**Mayank Bhagat**  
AI/ML Developer – CliniScan  

---

## ⭐ Contribute

- Fork the repo  
- Raise issues  
- Submit pull requests  

---

## 📜 License

MIT License
