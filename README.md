# Brain-cancer-prediction-app
## Overview
The Brain Cancer Prediction app is a machine learning-powered tool designed to assist medical professionals in detecting brain cancer. Using a set of measurements, the app predicts whether a cell mass is benign or malignant. It provides a visual representation of the input data using a radar chart and displays the predicted diagnosis and probability of being benign or malignant. The app can be used by manually inputting the measurements or by connecting it to a cytology lab to obtain the data directly from a machine. The connection to the laboratory machine is not a part of the app itself.
## Installation
To run the Cell Image Analyzer locally, you will need to have Python 3.6 or higher installed. Then, you can install the required packages by running:

```bash
pip install -r requirements.txt
```
This will install all the necessary dependencies, including Streamlit, OpenCV, and scikit-learn.

## Usage
To start the app, simply run the following command:

```bash
streamlit run app.py
```

This will launch the app in your default web browser. You can then upload an image of cells to analyze and adjust the various settings to customize the analysis. Once you are satisfied with the results, you can export the measurements to a CSV file for further analysis.

## Limitations

- The application includes only static data for training the machine learning model.
	> For learning purposes any individual can take the help of this app and can use backend support for dynamic data while integrating DB sources.

- To access the Measurement Guide, please Download the PDF and use the absolute path of your PDF in the source code of app.py

## Screenshots
