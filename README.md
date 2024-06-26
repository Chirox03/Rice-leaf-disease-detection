# Rice Leaf Disease Detection

This repository contains the code and models for the Rice Leaf Disease Detection project. The trained models are available for use with a simple Flask web application to demonstrate their functionality.

## Trained Models

The trained models used for this project can be found in the [Google Drive](https://drive.google.com/drive/u/0/folders/1m_VP28dfTR8tUL-TVvN0d_BWkZFyS_RQ).

## Flask Web Application

The Flask web application provides a simple interface to upload an image and get predictions for rice leaf diseases.

### Getting a Prediction

To get a prediction on an image using the trained models, use the following command:

```bash
python model.py -image_path="YOUR_IMAGE_PATH"
```
### Run the Flask demo app
1. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```
2. Install dependencies
```bash
pip install -r requirements.txt
python download_models.py
```
3. Start the Flask server
```bash
python app.py
```
