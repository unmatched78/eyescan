import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from fpdf import FPDF
import cv2
import pandas as pd
from io import BytesIO
from datetime import datetime
import os
import tempfile

# Configuration
MODEL_PATH = "retinal_model.keras"
CLASS_NAMES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
IMG_SIZE = (512, 512)
LANGUAGES = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr'
}

# Complete Translations
TRANSLATIONS = {
    'en': {
        'title': "Retinal Disease Classifier 👁️",
        'upload': "Upload retinal image(s)",
        'report_title': "Retinal Health Report",
        'diagnosis': "Diagnosis",
        'treatment': "Recommended Treatments",
        'batch_summary': "Batch Results Summary",
        'original': "Original Image",
        'attention': "Attention Map",
        'confidence': "Confidence (%)",
        'download_pdf': "Download PDF Report",
        'download_csv': "Export Batch Results"
    },
    'es': {
        'title': "Clasificador de Enfermedades Retinianas 👁️",
        'upload': "Subir imagen(es) retinales",
        'report_title': "Informe de Salud Retiniana",
        'diagnosis': "Diagnóstico",
        'treatment': "Tratamientos Recomendados",
        'batch_summary': "Resumen de Resultados",
        'original': "Imagen Original",
        'attention': "Mapa de Atención",
        'confidence': "Confianza (%)",
        'download_pdf': "Descargar PDF",
        'download_csv': "Exportar Resultados"
    },
    'fr': {
        'title': "Classificateur de Maladies Rétiniennes 👁️",
        'upload': "Télécharger image(s) rétinienne(s)",
        'report_title': "Rapport de Santé Rétinienne",
        'diagnosis': "Diagnostic",
        'treatment': "Traitements Recommandés",
        'batch_summary': "Résumé des Résultats",
        'original': "Image Originale",
        'attention': "Carte d'Attention",
        'confidence': "Confiance (%)",
        'download_pdf': "Télécharger PDF",
        'download_csv': "Exporter Résultats"
    }
}

# Medical Information
disease_info = {
    'Cataract': {
        'treatment': ['Surgery', 'Prescription glasses']
    },
    'Diabetic Retinopathy': {
        'treatment': ['Anti-VEGF injections', 'Laser treatment']
    },
    'Glaucoma': {
        'treatment': ['Eye drops', 'Laser therapy']
    },
    'Normal': {
        'treatment': ['Routine checkups']
    }
}

# Recursive helper function to find the last convolutional layer in the model,
# including nested models or sequential containers.
def find_last_conv_layer(model):
    # Iterate over layers in reverse order
    for layer in reversed(model.layers):
        # If the layer is a Conv2D layer, return it.
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        # If the layer itself is a model or has sublayers, search recursively.????
        if hasattr(layer, 'layers'):
            nested_layer = find_last_conv_layer(layer)
            if nested_layer:
                return nested_layer
    return None

# Updated Grad-CAM Implementation using the recursive search
def make_gradcam_heatmap(img_array, model):
    try:
        # "Warm up" the model by calling it so that all nested layers are built.
        _ = model(img_array)
        
        # Recursively find the last convolutional layer
        last_conv_layer = find_last_conv_layer(model)
        if not last_conv_layer:
            raise ValueError("No convolutional layer found in the model")
        
        # Create a gradient model that outputs the feature maps and predictions.
        grad_model = tf.keras.models.Model(
            [model.input],
            [last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]
        
        # Calculate gradients and pooled gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Create the heatmap
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)  # Avoid division by zero
        return heatmap[0]
    
    except Exception as e:
        st.error(f"Grad-CAM failed: {str(e)}")
        return np.zeros(IMG_SIZE)  # Return blank heatmap on error

# Fixed PDF Report Generation
def create_pdf_report(prediction_data, image_file, language='en'):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    try:
        if os.path.exists("medical_logo.png"):
            pdf.image("medical_logo.png", x=10, y=8, w=30)
    except Exception as e:
        st.error(f"Logo loading error: {str(e)}")
    
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, txt=TRANSLATIONS[language]['report_title'], ln=1)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=1)
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            Image.open(image_file).save(tmp.name)
            pdf.image(tmp.name, x=25, y=40, w=160)
        os.unlink(tmp.name)
    except Exception as e:
        st.error(f"PDF image error: {str(e)}")
    
    pdf.ln(10)
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, txt=f"{TRANSLATIONS[language]['diagnosis']}: {prediction_data['class']}", ln=1)
    pdf.cell(200, 10, txt=f"{TRANSLATIONS[language]['confidence']}: {prediction_data['confidence']:.1f}%", ln=1)
    pdf.multi_cell(0, 10, txt=f"{TRANSLATIONS[language]['treatment']}: {', '.join(prediction_data['treatment'])}")
    
    pdf_bytes = BytesIO()
    pdf_bytes.write(pdf.output(dest='S').encode('latin-1'))
    pdf_bytes.seek(0)
    return pdf_bytes

# Streamlit App
def main():
    lang = st.sidebar.selectbox("🌐 Language", list(LANGUAGES.keys()))
    lang_code = LANGUAGES[lang]
    st.title(TRANSLATIONS[lang_code]['title'])
    
    uploaded_files = st.file_uploader(
        TRANSLATIONS[lang_code]['upload'],
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        model = load_model()
        predictions = []
        
        for uploaded_file in uploaded_files:
            with st.expander(f"Analysis for {uploaded_file.name}"):
                image = Image.open(uploaded_file).convert('RGB')
                img_array = preprocess_image(image)
                
                pred = model.predict(img_array, verbose=0)
                confidence = tf.nn.softmax(pred[0]).numpy()
                pred_class = CLASS_NAMES[np.argmax(confidence)]
                
                try:
                    heatmap = make_gradcam_heatmap(img_array, model)
                    heatmap = cv2.resize(heatmap, IMG_SIZE)
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    
                    superimposed_img = cv2.addWeighted(
                        cv2.cvtColor(np.array(image.resize(IMG_SIZE)), cv2.COLOR_RGB2BGR),
                        0.6, heatmap, 0.4, 0
                    )
                except Exception as e:
                    st.error(f"Attention map error: {str(e)}")
                    superimposed_img = np.array(image.resize(IMG_SIZE))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption=TRANSLATIONS[lang_code]['original'])
                    st.pyplot(display_confidence_chart(confidence, lang_code))
                with col2:
                    st.image(superimposed_img, caption=TRANSLATIONS[lang_code]['attention'])
                    
                    pdf_data = create_pdf_report({
                        'class': pred_class,
                        'confidence': np.max(confidence)*100,
                        'treatment': disease_info[pred_class]['treatment']
                    }, uploaded_file, lang_code)
                    
                    st.download_button(
                        label=f"📄 {TRANSLATIONS[lang_code]['download_pdf']}",
                        data=pdf_data,
                        file_name=f"eye_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
                
                predictions.append({
                    'file': uploaded_file.name,
                    'prediction': pred_class,
                    'confidence': f"{np.max(confidence)*100:.1f}%"
                })
        
        st.subheader(TRANSLATIONS[lang_code]['batch_summary'])
        st.dataframe(pd.DataFrame(predictions))
        
        csv = pd.DataFrame(predictions).to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📊 {TRANSLATIONS[lang_code]['download_csv']}",
            data=csv,
            file_name="batch_results.csv",
            mime="text/csv"
        )

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    _ = model(tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3)))
    return model

def preprocess_image(image):
    img = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.image.per_image_standardization(img_array)
    return tf.expand_dims(img_array, 0)

def display_confidence_chart(confidence, lang_code):
    fig, ax = plt.subplots()
    ax.barh(CLASS_NAMES, confidence * 100)
    ax.set_xlabel(TRANSLATIONS[lang_code]['confidence'])
    ax.set_title(TRANSLATIONS[lang_code]['diagnosis'])
    return fig

if __name__ == "__main__":
    main()
