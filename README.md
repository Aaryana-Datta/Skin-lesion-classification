# DermAI — Skin Lesion Classification with EfficientNet-B3

This project trains a deep-learning model to classify seven skin lesion types from the HAM10000 (ISIC) dataset. I built the pipeline end-to-end in Google Colab: dataset preparation, training, evaluation, and Grad-CAM visualisation.

The goal was to develop a clean, well-structured medical-imaging project suitable for a portfolio.

---

## Dataset
I used the **HAM10000** dermatoscopic dataset. It contains 10,015 images and seven diagnostic classes:

- akiec – Actinic keratoses  
- bcc – Basal cell carcinoma  
- bkl – Benign keratosis  
- df – Dermatofibroma  
- nv – Melanocytic nevi  
- mel – Melanoma  
- vasc – Vascular lesions  

I created a stratified split:
- **80% train**
- **10% validation**
- **10% test**

Images were loaded from the two HAM10000 folders and linked via a custom `metadata.csv`.

---

## Model
I used **EfficientNet-B3**, pretrained on ImageNet, and fine-tuned it on 224×224 images.  
Key choices:

- AdamW optimizer  
- Cosine annealing learning-rate schedule  
- Cross-entropy loss with class-weighting to reduce imbalance effects  
- Mixed-precision training (AMP)  
- Minimal, safe augmentations (resize + horizontal flip)

---

## Results

### **Test Accuracy:** **82.04%**

### **Macro F1-score:** **0.7258**  
### **Weighted F1-score:** **0.8238**

The model performs best on the large “nv” class and on “vasc”. Performance drops on rare classes (0, 3, 5), which is expected due to heavy imbalance.

### Confusion Matrix
(Example values shown from final run)

- Strong diagonal for classes 4 (nv) and 6 (vasc)  
- Some confusion between:
  - benign keratosis (2) vs. melanoma (5)
  - melanoma (5) vs. basal cell carcinoma (1)
  - actinic keratoses (0) vs. benign keratosis (2)

This behaviour aligns with known clinical similarities between these lesions.

---

## Explainability (Grad-CAM)
I generated Grad-CAM heatmaps on test images.  
The overlays show that the model generally focuses on the central lesion region rather than background skin, which increases trust in the predictions.

---

## What I Learned
- How to handle medical image imbalance with class weights  
- How to fine-tune EfficientNet models efficiently  
- How to validate models properly using F1, confusion matrix and ROC-AUC  
- How to interpret CNN decisions using Grad-CAM  

---

## Limitations
This model is **not suitable for clinical use**.  
HAM10000 has:
- dataset imbalance  
- limited device variability  
- no patient-level splits  

Real medical deployment would require calibration, multi-center data, and dermatology oversight.

---

