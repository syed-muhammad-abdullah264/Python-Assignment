# Multi Animal Images Classification CNN

This project builds and runs an image classification app for five animal classes:
- cat
- cow
- deep
- dog
- lion

## Project files
- `app.py` — Streamlit web app
- `convert_from_scratch_with_augmentation.keras` — model used by the app
- `requirements.txt` — project dependencies

## Setup

Use Python 3.10, 3.11, or 3.12. TensorFlow is not compatible with
Python 3.14. Create a fresh virtual environment:

```bash
cd /path/to/Multi_Animal_Images_Classification_CNN
python -m venv venv
source venv/bin/activate
```

On Windows, activate it with `venv\Scripts\activate` instead.

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit in the browser.

## Notes
- The app loads `convert_from_scratch_with_augmentation.keras` from the project directory.
- The model expects RGB images and resizes them to its saved input size automatically.
- The app validates the model shape and gives a readable Streamlit error if a dependency or model file is missing.
