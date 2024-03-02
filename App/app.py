import streamlit as st
import pickle as pickle
import pandas as pd
from PIL import Image
from streamlit.components.v1 import html
import webbrowser
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def openPdf():
    pdf_path = os.path.abspath(
        "d:\Python Projects\Brain_cancer_prediction_app\App\Measurement Guide.pdf")
    webbrowser.open('file://' + pdf_path, new=2)
    

def clean_data():
    data = pd.read_csv('data/data.csv')

    # Remove the id and Unnamed column as those are unnecessary
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    # Map the diagnosis target variable to a binary classification format i.e 0 and 1
    data['diagnosis'] = data['diagnosis'].map({
        'M': 1,
        'B': 0
    })

    return data


def sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    # print(input_dict)
    return input_dict


def get_scaled_values(input_dict):
    data = clean_data()

    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def get_binary_scaled_values(input_dict):
    data = clean_data()

    X = data.drop(['diagnosis'], axis=1)
    max_limits = {}

    for key in input_dict.keys():
        max_val = X[key].max()
        max_limits[key] = max_val

    scaled_data = {key: value / max_limits[key]
                   for key, value in input_dict.items()}

    return scaled_data


def get_radar_chart_data(input_data):

    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter',
                  'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave points', 'Symmetry', 'Fractal dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'], input_data['area_mean'], input_data['smoothness_mean'], input_data[
                'compactness_mean'], input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'], input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'], input_data['smoothness_se'], input_data[
            'compactness_se'], input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'], input_data['area_worst'], input_data['smoothness_worst'], input_data[
                'compactness_worst'], input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'], input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        width=500,
        height=600,
        paper_bgcolor='#061634',
        legend_font_size=15,
        legend_borderwidth=30
    )

    return fig


def get_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>",
                 unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>",
                 unsafe_allow_html=True)

    st.write("This app can assist medical professionals in making a diagnosis choice, but should not be used as a substitute for a professional diagnosis.")


def get_pie_chart_data(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)
    # print(input_array_scaled)
    value_benign = round(model.predict_proba(input_array_scaled)[0][0], 3)
    value_malicious = round(model.predict_proba(input_array_scaled)[0][1], 3)

    labels = ['Benign', 'Malicious']

    values = [value_benign, value_malicious]
    colors = ['#ef553b', '#636efa']
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

    fig.update_traces(textfont_size=18,
                      marker=dict(colors=colors))
    return fig


def get_bar_chart_data(input_data):

    input_data = get_binary_scaled_values(input_data)
    fields_x = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness',
                'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']

    fig = go.Figure(data=[
        go.Bar(name='Mean', x=fields_x, y=[
               input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'], input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'], input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'], input_data['fractal_dimension_mean']], marker_color='rgb(229, 130, 3)'),
        go.Bar(name='Standard Error', x=fields_x, y=[
               input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']], marker_color='rgb(89, 231, 156 )'),
        go.Bar(name='Worst', x=fields_x, y=[
               input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'], input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'], input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'], input_data['fractal_dimension_worst']], marker_color='rgb(186, 130, 208)')
    ],
    )
    fig.update_layout(
        barmode='group',
        paper_bgcolor='#061634',
        height=450,
        width=350,
        margin=dict(l=100, r=150, t=50, b=100),
        # title_x=1,
        # title_standoff=25,
        legend_title_font_size=10,
        # title_font=dict(size=24),
        yaxis=dict(
            title='Logistic  Value',
            titlefont_size=18,
            tickfont_size=16,
            title_standoff=35
        ),

        bargap=0.25,
        bargroupgap=0.1,
        xaxis=dict(
            title='Parameters',
            titlefont_size=18,
            tickfont_size=14,
            title_standoff=35
        ),
    )

    return fig


def main():
    logo = Image.open(r'App\Images\logo.png')
    st.set_page_config(
        page_title="Brain Cancer Predictor",
        layout="wide",
        page_icon=logo,
        initial_sidebar_state="expanded"
    )

    # add styles
    with open("assests/style.css") as file:
        st.markdown("<style>{}</style>".format(file.read()),
                    unsafe_allow_html=True)

    # add sidebar
    input_dict = sidebar()
    # scaled_input_dict
    with st.container():
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.title("Brain Cancer Detection")
            st.markdown("Empowering Precision Medicine, leverage the power of AI-driven algorithms to predict malignant detection. Input tissue sample details to receive accurate brain cancer diagnoses. Facilitating early detection of cell mass as benign or malignant.")
            btn1 = st.button(label='Measurement Guide')

        with col2:
            image = Image.open(
                r"App\images\pic.jpg")
            st.image(image)

    if (btn1):
        st.session_state.runpage = openPdf
        st.session_state.runpage()

    with st.container():
        tab1, tab2 = st.tabs(["Charts", "How it works"])

        with tab1:
            col1, col2 = st.columns([2, 1], gap="large")
            with col1:
                st.subheader('Radar Chart')
                st.markdown("\n")
                radar_chart = get_radar_chart_data(input_dict)
                st.plotly_chart(radar_chart, use_container_width=True)
            with col2:
                st.subheader('Cell cluster Prediction')
                st.markdown("\n")
                get_predictions(input_dict)
                st.markdown("\n")
                pie_chart = get_pie_chart_data(input_dict)
                st.plotly_chart(pie_chart, use_container_width=True)

            st.subheader('Parameter Analysis')
            st.markdown("\n")
            bar_chart = get_bar_chart_data(input_dict)
            st.plotly_chart(bar_chart, use_container_width=True)

        with tab2:
            st.markdown("\n")
            st.write("Using a set of measurements, the app predicts whether a brain cell mass is benign or malignant. It provides a visual representation of the input data using a radar chart and displays the predicted diagnosis and probability of being benign or malignant. The app can be used by manually inputting the measurements or by connecting it to a cytology lab to obtain the data directly from a machine. The connection to the laboratory machine is not a part of the app itself.")
            st.markdown("\n")
            st.subheader("Working of the Model")
            st.markdown("\n")
            col1, col2 = st.columns([2, 1])
            with col1:
                image = Image.open(r"App/images/logic.jpg")
                st.image(image, width=800, use_column_width=True)
            with col2:
                st.write("For n no of inputs the model will predict a binary classified output as Malignant or Benign. As per Logistic Regression algorithm, the model will place each data point value to either 0 or 1 using nearest assertion. ")
                st.markdown("""
                    Here are the usabilities of the Logistic Regression:

                    - It uses the sigmoid function that is used to map the predictive values to probabilities
                    - Dependent variable (Diagnosis) is categorical in nature.
                    - Independent variable (Radius, Perimeter, Concavity, .. etc.) don't have any multi-collinearity.
                    """)

            st.markdown("\n")
              


if __name__ == '__main__':
    main()
