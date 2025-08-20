# Severstal_Steel_Defect_Detection

# 🛠️ ML & AI Internship Assignment – Defect Classification

## 📌 Objective
The task was to classify industrial equipment images into:
- **Defective**
- **Non-Defective**

### 🔹 Bonus Objectives
- Identify and classify specific defect types.
- Optimize model for hardware-accelerated inference (GPU, TPU, Jetson Nano).

---

## ⚡ Approach
I used **YOLOv5** for defect classification instead of a traditional CNN.  
- YOLO allows **real-time classification & localization**.  
- CNN-based approaches would take longer to train on large-scale data.  
- Trained on local GPU instead of cloud due to dataset size and constraints.  

---

## 🧩 Challenges Faced
- **Large dataset size** → caused storage & training time constraints.  
- **Limited compute** → had to rely on a local GPU instead of cloud.  
- **Image resolution** → original resolution was `256x1600`, needed resizing (`--img 640`) for YOLOv5 while maintaining correct label scaling.  

---

## 🚀 Results
- Trained YOLOv5 model (`yolov5s.pt` fine-tuned).  
- Model distinguishes between **Defective** and **Non-Defective** with promising accuracy.  
- Bonus: The `.pt` model can be converted to **TensorRT engine** for **Jetson Nano deployment** with CUDA acceleration.  

---

## 🔧 Installation
```bash
git clone [https://github.com/<your-username>/ML-AI-Defect-Classification.git](https://github.com/akhileshshinde/Severstal_Steel_Defect_Detection.git)
pip install -r requirements.txt
