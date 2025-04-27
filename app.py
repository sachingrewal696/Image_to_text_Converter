import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import os

# Load the model (new .keras format)
@st.cache_resource
def load_caption_model():
    model = load_model('model.keras')
    return model

model = load_caption_model()

# Load supporting files
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('mapping.pkl', 'rb') as f:
    mapping = pickle.load(f)

# Helper functions
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image_feature, tokenizer, max_length=35):  # Use max_length=35 (model's expected value)
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')[0]

        # Ensure the image feature has the correct shape (1, 4096)
        image_feature = np.squeeze(image_feature)  # Remove extra dimensions (if any)
        
        if image_feature.ndim == 1:  # Check if it's a 1D array with shape (4096,)
            image_feature = np.expand_dims(image_feature, axis=0)  # Reshape to (1, 4096)
        
        yhat = model.predict([image_feature, np.expand_dims(sequence, axis=0)], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

# Streamlit app starts here
st.title("🖼️ Image Caption Generator")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Extract image ID
    image_name = uploaded_file.name
    image_id = os.path.splitext(image_name)[0]

    if image_id in features:
        st.write("Generating Caption... Please wait ⏳")
        max_length = 35  # Set to 35 to match the model's expected sequence length
        y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
        st.subheader("📝 Predicted Caption:")
        st.success(y_pred.replace('start', '').replace('end', '').strip())
    else:
        st.error("❌ Feature for this image is not available. Please use a flicker dataset image.")
