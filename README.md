
# 🧨 Adversarial_Attack_OCR

This repository demonstrates adversarial attacks on an OCR model using **Projected Gradient Descent (PGD)**. The goal is to evaluate and reduce the robustness of OCR systems by generating adversarial perturbations using three different strategies:

- **Global PGD**
- **HF-Focused PGD**
- **Stroke-Focused PGD**

## 📁 Repository Structure

```
Adversarial_Attack_OCR/
├── data/Train/             # Training dataset (images + labels)
├── train_vietocr.py        # Script to train the VietOCR model
├── attack_PGD.py           # PGD attack script for Global and HF-Focused variants
├── Stroke_focused.py       # PGD attack script for Stroke-Focused variant
├── README.md               # Project documentation
```

## 🧪 Attack Variants

### 1. Global PGD

- Implemented in: `attack_PGD.py`
- Perturbs the **entire image uniformly**
- Serves as the **baseline** PGD attack

### 2. HF-Focused PGD

- Also in: `attack_PGD.py`
- Perturbs only **high-frequency regions** such as edges and textures
- Requires setting `"hf_focus"` inside the script

### 3. Stroke-Focused PGD

- Implemented in: `Stroke_focused.py`
- Perturbs only the **character stroke regions**
- Designed for more **localized and stealthy** adversarial perturbations

> 🔁 To switch between Global and HF-Focused PGD, edit the `attack_mode` variable inside `attack_PGD.py`:

```python
# Inside attack_PGD.py
attack_mode = "global"  # or "hf_focus"
```

## 🚀 Usage Instructions

### 1. Install Dependencies

```bash
pip install torch torchvision vietocr opencv-python numpy
```

### 2. Train the OCR Model

```bash
python train_vietocr.py
```

> You may need to modify data paths inside the script depending on your setup.

### 3. Run Global or HF-Focused PGD Attack

```bash
python attack_PGD.py
```

- Make sure to set the desired `attack_mode` inside the script (`"global"` or `"hf_focus"`)

### 4. Run Stroke-Focused PGD Attack

```bash
python Stroke_focused.py
```

## ⚙️ Attack Parameters

Each PGD attack can be configured with the following hyperparameters:

- `epsilon`: Maximum perturbation magnitude (L∞ norm)
- `steps`: Number of PGD iterations
- `step-size`: Step size for each PGD step

> You can either pass these via CLI (if implemented) or modify directly inside the script.

## 📊 Evaluation Metrics (Optional)

If you wish to evaluate the impact of attacks on model performance:

- **Character Error Rate (CER)**
- **Word Error Rate (WER)**
- **Sequence Error Rate (SER)**

These can be computed by comparing predictions on clean vs. adversarial images.

## 📌 Notes

- All attacks are **white-box**, using gradients from the trained OCR model.
- The OCR model is based on [VietOCR](https://github.com/quanpn90/VietOCR).
- Dataset used: Vietnamese printed or handwritten text samples under `data/Train/`.

## 👤 Author

**tbaro19**
