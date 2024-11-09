import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf
import logging


#load plant_classification model
MODEL = tf.keras.models.load_model("./model.h5")

#plant_classification_classes
#CLASS_NAMES = ['Apple_Plant','Bell_Plant','Citrus_Plant','Grape_Plant','Maize_Plant','Peach_Plant','Potato_Plant','Strawberry_Plant','Tomato_Plant']
CLASS_NAMES = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry___including_sour___Powdery_mildew',
 'Cherry___including_sour___healthy',
 'Corn___maize___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___maize___Common_rust_',
 'Corn___maize___Northern_Leaf_Blight',
 'Corn___maize___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper___bell___Bacterial_spot',
 'Pepper___bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

#====================
plant_diseases_info = [
    {
        "disease": "Apple___Apple_scab",
        "cause": "Fungus Venturia inaequalis.",
        "prevention": "Use resistant varieties, ensure good air circulation, and prune infected branches.",
        "medication": "Apply fungicides like captan or mancozeb during the growing season."
    },
    {
        "disease": "Apple___Black_rot",
        "cause": "Fungus Botryosphaeria obtusa.",
        "prevention": "Remove and destroy infected fruits and leaves; prune cankers.",
        "medication": "Fungicides such as thiophanate-methyl can be effective."
    },
    {
        "disease": "Apple___Cedar_apple_rust",
        "cause": "Fungus Gymnosporangium juniperi-virginianae.",
        "prevention": "Remove nearby juniper trees; use resistant apple varieties.",
        "medication": "Apply fungicides such as myclobutanil or propiconazole."
    },
    {
        "disease": "Apple___healthy",
        "status": "healthy"
    },
    {
        "disease": "Blueberry___healthy",
        "status": "healthy"
    },
    {
        "disease": "Cherry___(including_sour)___Powdery_mildew",
        "cause": "Fungus Podosphaera clandestina.",
        "prevention": "Ensure proper spacing and sunlight; prune to increase air flow.",
        "medication": "Use sulfur-based fungicides or potassium bicarbonate."
    },
    {
        "disease": "Cherry___(including_sour)___healthy",
        "status": "healthy"
    },
    {
        "disease": "Corn___(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "cause": "Fungus Cercospora zeae-maydis.",
        "prevention": "Rotate crops; plant resistant hybrids.",
        "medication": "Apply fungicides like strobilurins or triazoles if necessary."
    },
    {
        "disease": "Corn___(maize)___Common_rust_",
        "cause": "Fungus Puccinia sorghi.",
        "prevention": "Use resistant varieties; remove debris from previous crops.",
        "medication": "Fungicides such as propiconazole or azoxystrobin may help control severe outbreaks."
    },
    {
        "disease": "Corn___(maize)___Northern_Leaf_Blight",
        "cause": "Fungus Exserohilum turcicum.",
        "prevention": "Choose resistant varieties; practice crop rotation.",
        "medication": "Fungicides like mancozeb or chlorothalonil can be used if needed."
    },
    {
        "disease": "Corn___(maize)___healthy",
        "status": "healthy"
    },
    {
        "disease": "Grape___Black_rot",
        "cause": "Fungus Guignardia bidwellii.",
        "prevention": "Prune to improve air circulation; remove infected leaves and berries.",
        "medication": "Fungicides like mancozeb or captan can help control the disease."
    },
    {
        "disease": "Grape___Esca_(Black_Measles)",
        "cause": "Complex of fungi including Phaeomoniella chlamydospora and Phaeoacremonium aleophilum.",
        "prevention": "Avoid injuries to the vines; maintain proper pruning techniques.",
        "medication": "No effective chemical treatment; focus on cultural practices."
    },
    {
        "disease": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "cause": "Fungus Pseudocercospora vitis.",
        "prevention": "Ensure good air circulation; remove infected plant material.",
        "medication": "Fungicides such as copper-based products can help manage the disease."
    },
    {
        "disease": "Grape___healthy",
        "status": "healthy"
    },
    {
        "disease": "Orange___Haunglongbing_(Citrus_greening)",
        "cause": "Bacterium Candidatus Liberibacter asiaticus, spread by the Asian citrus psyllid.",
        "prevention": "Control the Asian citrus psyllid vector; use certified disease-free planting material.",
        "medication": "No cure; remove and destroy infected trees to prevent spread."
    },
    {
        "disease": "Peach___Bacterial_spot",
        "cause": "Bacterium Xanthomonas arboricola pv. pruni.",
        "prevention": "Use resistant varieties; avoid overhead irrigation.",
        "medication": "Copper-based sprays can reduce severity but won't eliminate the disease."
    },
    {
        "disease": "Peach___healthy",
        "status": "healthy"
    },
    {
        "disease": "Pepper___bell___Bacterial_spot",
        "cause": "Bacterium Xanthomonas campestris pv. vesicatoria.",
        "prevention": "Use disease-free seeds; avoid working in wet conditions.",
        "medication": "Copper sprays can help manage the disease but are not a cure."
    },
    {
        "disease": "Pepper___bell___healthy",
        "status": "healthy"
    },
    {
        "disease": "Potato___Early_blight",
        "cause": "Fungus Alternaria solani.",
        "prevention": "Rotate crops; use certified seed potatoes.",
        "medication": "Apply fungicides such as chlorothalonil or mancozeb."
    },
    {
        "disease": "Potato___Late_blight",
        "cause": "Oomycete Phytophthora infestans.",
        "prevention": "Plant resistant varieties; avoid wet foliage.",
        "medication": "Fungicides such as metalaxyl or chlorothalonil are effective."
    },
    {
        "disease": "Potato___healthy",
        "status": "healthy"
    },
    {
        "disease": "Raspberry___healthy",
        "status": "healthy"
    },
    {
        "disease": "Soybean___healthy",
        "status": "healthy"
    },
    {
        "disease": "Squash___Powdery_mildew",
        "cause": "Fungi in the genera Podosphaera or Erysiphe.",
        "prevention": "Ensure good air circulation; water at the base of plants.",
        "medication": "Use sulfur or potassium bicarbonate sprays to control."
    },
    {
        "disease": "Strawberry___Leaf_scorch",
        "cause": "Fungus Diplocarpon earlianum.",
        "prevention": "Avoid overhead irrigation; maintain good air circulation.",
        "medication": "Fungicides such as captan can help manage the disease."
    },
    {
        "disease": "Strawberry___healthy",
        "status": "healthy"
    },
    {
        "disease": "Tomato___Bacterial_spot",
        "cause": "Bacterium Xanthomonas campestris pv. vesicatoria.",
        "prevention": "Use certified seeds; avoid working with wet plants.",
        "medication": "Copper sprays may help reduce disease severity."
    },
    {
        "disease": "Tomato___Early_blight",
        "cause": "Fungus Alternaria solani.",
        "prevention": "Remove infected leaves; use mulches to reduce soil splash.",
        "medication": "Fungicides such as chlorothalonil or copper-based products can be used."
    },
    {
        "disease": "Tomato___Late_blight",
        "cause": "Oomycete Phytophthora infestans.",
        "prevention": "Plant resistant varieties; avoid overhead watering.",
        "medication": "Fungicides like mancozeb or copper can help manage the disease."
    },
    {
        "disease": "Tomato___Leaf_Mold",
        "cause": "Fungus Passalora fulva.",
        "prevention": "Ensure good air circulation; avoid high humidity.",
        "medication": "Use fungicides such as copper or mancozeb."
    },
    {
        "disease": "Tomato___Septoria_leaf_spot",
        "cause": "Fungus Septoria lycopersici.",
        "prevention": "Remove lower leaves; avoid watering from above.",
        "medication": "Fungicides like chlorothalonil can be effective."
    },
    {
        "disease": "Tomato___Spider_mites Two-spotted_spider_mite",
        "cause": "Spider mites Tetranychus urticae.",
        "prevention": "Maintain humidity and regular watering; avoid dusty conditions.",
        "medication": "Use miticides like abamectin or insecticidal soap."
    },
    {
        "disease": "Tomato___Target_Spot",
        "cause": "Fungus Corynespora cassiicola.",
        "prevention": "Ensure good air circulation; remove infected leaves.",
        "medication": "Apply fungicides such as chlorothalonil."
    },
    {
        "disease": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "cause": "Tomato yellow leaf curl virus, transmitted by whiteflies.",
        "prevention": "Control whiteflies; use resistant varieties.",
        "medication": "No chemical treatments available; focus on prevention."
    },
    {
        "disease": "Tomato___Tomato_mosaic_virus",
        "cause": "Tomato mosaic virus.",
        "prevention": "Use certified seeds; practice good sanitation.",
        "medication": "No treatment; remove infected plants and control aphids."
    },
    {
        "disease": "Tomato___healthy",
        "status": "healthy"
    }
]




# Set the page configuration
st.set_page_config(page_title="Simple Image Uploader", layout="wide")

# Sidebar for file upload
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def resize_image(image, max_width=300):
    """Resize the image to a maximum width while maintaining the aspect ratio."""
    width, height = image.size
    if width > max_width:
        new_height = int(max_width * height / width)
        return image.resize((max_width, new_height))
    return image

def get_plant_name(class_name):
    """Extract and return only the plant name from the class label."""
    return class_name.split('___')[0]


def get_condition_name(class_name):
    """Extract and return the text after '___' from the class label."""
    return class_name.split('___')[1] if '___' in class_name else 'Unknown' 


def find_disease_info(class_name):
    """Find the disease information from the plant_diseases_info list based on the class name."""
    for info in plant_diseases_info:
        if info.get("disease") == class_name:
            return info
    return None
# Display the uploaded image
if uploaded_file is not None:
    #image = Image.open(uploaded_file)
    
    def read_file_as_image(data) -> np.ndarray:
        image = np.array(Image.open(BytesIO(data)).convert("RGB"))
        resized_image = np.array(Image.fromarray(image).resize((224, 224)))
        logging.info("Resized Image Shape: %s", resized_image.shape)
        return resized_image
    
    image = read_file_as_image(uploaded_file.read())

    img_batch = tf.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    plant_name = get_plant_name(predicted_class)
    confidence = np.max(predictions[0])
    image = Image.open(uploaded_file)
    new_image = image.resize((300,300))
    condition_name = get_condition_name(predicted_class)
    resized_image = resize_image(image, max_width=200)  #
    col1, col2 = st.columns(2)

    with col1:
        # Display the image in the first column
        st.image(resized_image, caption="Uploaded Image", use_column_width=True)

    with col2:
        # Display the prediction details and disease information in the second column
        st.write(f"**Predicted Plant:** {plant_name}")
        st.write(f"**Condition:** {condition_name}")
        st.write(f"**Confidence:** {confidence:.2%}")

        # Fetch and display disease information
        disease_info = find_disease_info(predicted_class)
        if disease_info:
            st.write(f"**Cause:** {disease_info.get('cause', 'Information not available')}")
            st.write(f"**Prevention:** {disease_info.get('prevention', 'Information not available')}")
            st.write(f"**Medication:** {disease_info.get('medication', 'Information not available')}")
        else:
            st.write("No additional information available for this condition.")

else:
    st.write("Upload an image using the sidebar.")