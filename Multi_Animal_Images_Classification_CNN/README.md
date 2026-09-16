# Multi Animal Images Classification CNN

This project builds and runs an image classification app for five animal classes:
- cat
- cow
- deep
- dog
- lion

## Project files
- `app.py` — Streamlit web app
- `animal_classification_model.pkl` — trained model file used by the app
- `convert_from_scratch_with_augmentation.keras` — saved Keras model file
- `animals_dataset/` — dataset folder
- `requirements.txt` — project dependencies

## Setup

Create a virtual environment:

```powershell
cd "D:\Python\Assignment\Multi_Animal_Images_Classification_CNN"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

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
- The app uses the trained animal model stored in `animal_classification_model.pkl`.
- The project includes a Keras version as well for compatibility with TensorFlow-based workflows.
- The `.gitignore` excludes virtual environments, checkpoints, logs, and generated model files from version control.
