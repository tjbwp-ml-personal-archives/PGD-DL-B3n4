import streamlit as st
import numpy as np
from PIL import Image
import cv2
import wget

# Download Cascade classifier file only one time
#url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
#xml_file = wget.download(url)  # downloaded 'haarcascade_frontalface_default.xml'
xml_path = 'haarcascade_frontalface_default.xml'
# Initialize the cascade classifiers for face
face_cascade = cv2.CascadeClassifier(xml_path)

# Define a function to process the uploaded image and produce results
def process_image(image):
    # Read image
    img = image

    # Getting the detections
    detections = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
    
    result = False
    # Draw detections
    if len(detections) > 0:
        for face in detections:
            cv2.rectangle(img, face, (0, 255, 0), 5)
        result = True
    # Display image
    processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return result, processed_image, detections

# Streamlit app title and description
st.title("NED PGD - DL Class")
st.write("This Streamlit app allows you to upload an image and process it with a backend model.")

result = None
processed_image = None
detections = None

# Upload an image file
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    cv2_image = cv2.imdecode(file_bytes, 1)

    # Process the image and get the results
    result, processed_image, detections = process_image(cv2_image)

    # Display the processed image if available
    if result is not None:
        st.subheader("Processed Image")
        st.image(processed_image, caption="Processed Image", use_column_width=True)
        st.write(f"Total {len(detections)} person found in the image.")

        # Ask the user if they want to continue or quit
        user_choice = st.radio("What would you like to do next?", ("Continue", "Quit"))

        # If the user chooses to continue, clear the uploaded image and result
        if user_choice == "Continue":
            uploaded_image = None
            result = None
elif (uploaded_image and not processed_image):
    st.write("Can't process the image. Please upload a valid image.")
