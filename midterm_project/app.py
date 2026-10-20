# ----------------------------------
# IMPORTS AND MODEL LOADING
# ----------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import joblib

# Load dataset
df = pd.read_csv("cherry_features.csv")

# Load trained SVM model
model = joblib.load("svm_model.pkl")

# ----------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Dataset Exploration",
        "Feature Analysis",
        "Correlation Analysis",
        "Model Performance",
        "Live Prediction",
        "Conclusions"
    ]
)

# ==================================
# OVERVIEW
# ==================================

if page == "Overview":

    st.title("Cherry Leaf Powdery Mildew Detection")

    st.write("""
    This project uses image analysis and machine learning to classify
    cherry leaves as healthy or infected with powdery mildew.
    """)

    healthy_count = (df["label"] == "Healthy").sum()
    mildew_count = (df["label"] == "Powdery Mildew").sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Healthy Images", healthy_count)
    col2.metric("Diseased Images", mildew_count)
    col3.metric("Total Images", len(df))

    st.subheader("Dataset Composition")

    label_counts = df["label"].value_counts()

    fig = px.pie(
        values=label_counts.values,
        names=label_counts.index,
        title="Healthy vs Powdery Mildew"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Features Used")

    st.markdown("""
    - Average Red
    - Average Green
    - Average Blue
    - Brightness
    - Width
    - Height
    """)

# ==================================
# DATASET EXPLORATION
# ==================================

elif page == "Dataset Exploration":

    st.title("Dataset Exploration")

    filter_option = st.selectbox(
        "Filter by Class",
        ["All", "Healthy", "Powdery Mildew"]
    )

    if filter_option == "All":
        filtered_df = df
    else:
        filtered_df = df[df["label"] == filter_option]

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df)

    st.subheader("Dataset Shape")
    st.write(filtered_df.shape)

    st.subheader("Summary Statistics")
    st.dataframe(filtered_df.describe())

# ==================================
# FEATURE ANALYSIS
# ==================================

elif page == "Feature Analysis":

    st.title("Feature Analysis")

    feature = st.selectbox(
        "Choose Feature",
        [
            "avg_red",
            "avg_green",
            "avg_blue",
            "brightness"
        ]
    )

    st.subheader("Distribution")

    fig = px.histogram(
        df,
        x=feature,
        color="label",
        barmode="overlay",
        title=f"Distribution of {feature}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparison by Class")

    fig2 = px.box(
        df,
        x="label",
        y=feature,
        color="label",
        title=f"{feature} by Class"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    ### Interpretation

    This visualization compares the selected feature between healthy
    leaves and leaves infected with powdery mildew.

    Features showing greater separation between the groups may
    provide stronger predictive power for machine learning models.
    """)

# ==================================
# CORRELATION ANALYSIS
# ==================================

elif page == "Correlation Analysis":

    st.title("Correlation Analysis")

    corr = df[
        [
            "avg_red",
            "avg_green",
            "avg_blue",
            "brightness"
        ]
    ].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Viridis",
        title="Feature Correlation Heatmap"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Interpretation

    This heatmap shows the strength of relationships between image features.

    Values near 1 indicate strong positive relationships.

    The results suggest that brightness is strongly related to the
    average RGB values, which is expected because brighter images
    tend to contain higher overall color intensities.
    """)

# ==================================
# MODEL PERFORMANCE
# ==================================

elif page == "Model Performance":

    st.title("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        "74.9%"
    )

    col2.metric(
        "Precision",
        "76.0%"
    )

    col3.metric(
        "Recall",
        "75.0%"
    )

    col4.metric(
        "F1 Score",
        "74.0%"
    )

    st.subheader("Model Comparison")

    results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Support Vector Machine"
        ],
        "Accuracy": [
            71.5,
            72.5,
            74.9
        ]
    })

    st.dataframe(results)

    fig = px.bar(
        results,
        x="Model",
        y="Accuracy",
        color="Model",
        title="Model Accuracy Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.success(
        "Support Vector Machine achieved the highest accuracy (74.9%)."
    )

    st.subheader("Dataset Split")

    split_df = pd.DataFrame({
        "Dataset": [
            "Training",
            "Validation",
            "Testing"
        ],
        "Images": [
            1334,
            286,
            286
        ]
    })

    st.dataframe(split_df)

    st.subheader("Random Forest Confusion Matrix")

    st.image(
        "confusion_matrix.png",
        caption="Random Forest Confusion Matrix"
    )

    st.markdown("""
    ### Interpretation

    The Random Forest model correctly identified most
    powdery mildew infections.

    The model occasionally classified healthy leaves
    as infected, but it rarely missed infected leaves.

    This behavior is desirable in plant disease
    detection because it reduces the chance of
    overlooking a diseased plant.
    """)

# ==================================
# CONCLUSIONS
# ==================================

elif page == "Conclusions":

    st.title("Conclusions")

    st.markdown("""
    ## Summary

    This project demonstrates the application of machine learning techniques
    for detecting powdery mildew in cherry leaves.

    The analysis revealed that certain image features, particularly brightness
    and average RGB values, are strong indicators of leaf health.

    The trained Support Vector Machine (SVM) model achieved a test accuracy
    of 74.9%, indicating its effectiveness in classifying leaf conditions.

    ## Future Work

    - Collect more diverse datasets to improve model generalization.
    - Experiment with deep learning approaches for potentially better performance.
    - Implement real-time detection capabilities for practical applications.
    """)

# ==================================
# LIVE PREDICTION
# ==================================

elif page == "Live Prediction":

    st.title("Cherry Leaf Disease Detection System")

    st.markdown("""
    Upload a cherry leaf image to determine whether the leaf is healthy
    or infected with powdery mildew using a trained Support Vector
    Machine (SVM) model.
    """)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            width=350
        )

        # Feature extraction
        img_array = np.array(image)

        avg_red = np.mean(img_array[:, :, 0])
        avg_green = np.mean(img_array[:, :, 1])
        avg_blue = np.mean(img_array[:, :, 2])

        brightness = np.mean(
            img_array.mean(axis=2)
        )

        # Features used by trained model
        features = np.array([
            [
                avg_red,
                avg_green,
                avg_blue,
                brightness
            ]
        ])

        # Prediction
        prediction = model.predict(features)

        probabilities = model.predict_proba(features)[0]
        confidence = np.max(probabilities) * 100

        st.subheader("Prediction")

        if prediction[0] == 1:
            st.error("🌿 Powdery Mildew Detected")
        else:
            st.success("✅ Healthy Leaf")

        st.write(
            f"Confidence: {confidence:.2f}%"
        )

        st.progress(confidence / 100)

        st.subheader("Extracted Features")

        feature_df = pd.DataFrame({
            "Feature": [
                "Average Red",
                "Average Green",
                "Average Blue",
                "Brightness"
            ],
            "Value": [
                round(avg_red, 2),
                round(avg_green, 2),
                round(avg_blue, 2),
                round(brightness, 2)
            ]
        })

        st.dataframe(
            feature_df,
            use_container_width=True
        )

        st.markdown("""
        ### How Predictions Work

        The image is converted into numerical values by calculating:

        - Average Red intensity
        - Average Green intensity
        - Average Blue intensity
        - Overall Brightness

        These extracted features are passed into the trained Support
        Vector Machine (SVM), which predicts whether the leaf is
        Healthy or infected with Powdery Mildew.
        """)

        st.markdown("""
        ### Model Information

        **Model:** Support Vector Machine (SVM)

        **Training Images:** 1334

        **Validation Images:** 286

        **Testing Images:** 286

        **Test Accuracy:** 74.9%
        """)

# ==================================
# CONCLUSIONS
# ==================================

elif page == "Conclusions":

    st.title("Project Conclusions")

    st.markdown("""
    ## Summary

    This project developed a machine learning system for
    classifying cherry leaves as either Healthy or infected
    with Powdery Mildew.

    Image features were extracted from each leaf image,
    including average red, green, blue, and brightness
    values. These features were analyzed through
    exploratory data analysis and used to train machine
    learning models.
    """)

    st.markdown("""
    ## Key Findings

    - The dataset contained 1,906 cherry leaf images.
    - Healthy leaves and infected leaves showed measurable
      differences in color and brightness features.
    - Correlation analysis revealed strong relationships
      between brightness and RGB values.
    - Multiple machine learning models were evaluated,
      including Logistic Regression, Random Forest, and
      Support Vector Machine.
    - The Support Vector Machine achieved the highest
      classification accuracy of 74.9%.
    """)

    st.markdown("""
    ## Model Performance

    The final Support Vector Machine model achieved:

    - Accuracy: 74.9%
    - Precision: 76.0%
    - Recall: 75.0%
    - F1 Score: 74.0%

    These results demonstrate that simple image-derived
    color features can effectively distinguish between
    healthy leaves and leaves infected with powdery mildew.
    """)

    st.markdown("""
    ## Future Improvements

    Future work could improve performance by:

    - Incorporating texture-based image features.
    - Increasing the size of the training dataset.
    - Testing additional machine learning algorithms.
    - Implementing deep learning approaches such as
      Convolutional Neural Networks (CNNs).
    - Deploying the application to a cloud environment
      for public use.
    """)

    st.success(
        "The project successfully demonstrates an end-to-end machine learning workflow, including data exploration, feature engineering, model evaluation, and deployment through an interactive Streamlit application."
    )