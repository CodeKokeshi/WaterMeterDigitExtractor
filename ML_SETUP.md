# LeNet-5 Setup

These are the parts only you can do on your machine.

## 1. Install a compatible Python

TensorFlow for Windows does not work with the app's current Python 3.14 runtime.
Install one of these instead:

- Python 3.10
- Python 3.11
- Python 3.12
- Python 3.13

## 2. Create a dedicated ML environment

Example with Python 3.13:

```powershell
py -3.13 -m venv .venv-ml
.\.venv-ml\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r training_requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 3. Use that Python inside the app

In `Training > Train LeNet-5 Digit Model...`, set `Backend Python` to:

```text
<your-project>\.venv-ml\Scripts\python.exe
```

The same backend Python should also be used for testing.

## 4. Training workflow

1. Open `Training > Train LeNet-5 Digit Model...`
2. Select the dataset folder that contains `0` to `9`
3. Choose the TensorFlow/Keras output folder
4. Choose the TFLite output folder
5. Choose the backend Python
6. Start training

Outputs:

- `lenet5_digits.keras`
- `lenet5_digits.tflite`
- `labels.json`
- `metrics.json`

## 5. Testing workflow

1. Open `Testing > Select LeNet-5 Model...`
2. Pick the `.tflite` or `.keras` model
3. Enable `Testing > Enable Viewer Testing Mode`
4. Load an image in the main app
5. Select 4 points exactly like dataset creation
6. Click `Extract & Predict`

Optional:

- Type an expected 5-digit label in the label box before predicting
- The app will compare the prediction against it

## Notes

- Training runs in a background thread so the UI stays responsive.
- Viewer testing reuses the app's existing 4-point extraction pipeline.
- The current implementation is for LeNet-5 only. YOLOv8 can be added later on top of the same modular backend pattern.
