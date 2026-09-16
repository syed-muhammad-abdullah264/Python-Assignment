from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model as load_keras_model


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Animal Classification CNN",
    page_icon="🐾",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================


st.markdown("""
<style>

/* =====================================================
   GLOBAL APP
   ===================================================== */

.stApp {
    background:
        linear-gradient(
            135deg,
            #0f172a 0%,
            #172554 45%,
            #1e293b 100%
        ) !important;

    color: #f8fafc !important;
}


/* =====================================================
   ALL TEXT
   ===================================================== */

.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp div,
.stApp li {
    color: #e2e8f0;
}


/* =====================================================
   HEADINGS
   ===================================================== */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #f8fafc !important;
}


/* =====================================================
   TITLE
   ===================================================== */

.title {
    text-align: center;
    font-size: 45px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 5px;
    color: #f8fafc !important;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1 !important;
    margin-bottom: 35px;
}


/* =====================================================
   WHITE CARDS
   ===================================================== */

.card {
    padding: 25px;
    border-radius: 20px;

    background: #ffffff !important;

    border: 1px solid #e2e8f0;

    box-shadow:
        0 10px 35px rgba(0, 0, 0, 0.25);

    margin-bottom: 20px;
}


/* =====================================================
   TEXT INSIDE CARDS
   ===================================================== */

.card h1,
.card h2,
.card h3,
.card h4,
.card h5,
.card h6 {
    color: #0f172a !important;
}

.card p,
.card span,
.card label,
.card div {
    color: #334155;
}


/* =====================================================
   STREAMLIT SUBHEADER
   ===================================================== */

[data-testid="stSubheader"] {
    color: #0f172a !important;
}

[data-testid="stSubheader"] * {
    color: #0f172a !important;
}


/* =====================================================
   FILE UPLOADER
   ===================================================== */

[data-testid="stFileUploader"] {
    background: #f1f5f9 !important;

    border: 2px dashed #6366f1 !important;

    border-radius: 16px !important;

    padding: 12px !important;
}


/* Upload text */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] div {
    color: #334155 !important;
}


/* Browse button */
[data-testid="stFileUploader"] button {
    background: #4f46e5 !important;

    color: #ffffff !important;

    border: none !important;

    border-radius: 10px !important;
}

[data-testid="stFileUploader"] button * {
    color: #ffffff !important;
}


/* =====================================================
   PREDICT BUTTON
   ===================================================== */

.stButton > button {
    width: 100% !important;

    height: 52px !important;

    border-radius: 13px !important;

    border: none !important;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        ) !important;

    color: #ffffff !important;

    font-size: 16px !important;

    font-weight: 700 !important;

    box-shadow:
        0 8px 20px rgba(79, 70, 229, 0.35);

    transition: 0.25s ease;
}

.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #ffffff !important;
}

.stButton > button:hover {
    transform: translateY(-2px);

    background:
        linear-gradient(
            135deg,
            #4338ca,
            #6d28d9
        ) !important;
}


/* =====================================================
   RESULT CARD
   ===================================================== */

.result {
    padding: 30px;

    border-radius: 20px;

    background:
        linear-gradient(
            135deg,
            #ffffff,
            #eef2ff
        ) !important;

    border: 1px solid #c7d2fe;

    text-align: center;

    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.10);
}


/* Result text */
.result .animal {
    color: #ffffff !important;
}

.result .confidence {
    color: #4f46e5 !important;
}


/* =====================================================
   ANIMAL NAME
   ===================================================== */

.animal {
    font-size: 42px;

    font-weight: 800;

    margin: 10px 0;

    color: #ffffff !important;
}


/* =====================================================
   CONFIDENCE
   ===================================================== */

.confidence {
    font-size: 22px;

    font-weight: 700;

    color: #4f46e5 !important;
}


/* =====================================================
   PROBABILITY TEXT
   ===================================================== */

.stMarkdown p,
.stMarkdown strong {
    color: #334155 !important;
}


/* =====================================================
   INFO MESSAGE
   ===================================================== */

[data-testid="stAlert"] {
    background: #eff6ff !important;

    border: 1px solid #bfdbfe !important;

    border-radius: 14px !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #1e3a8a !important;
}


/* =====================================================
   PROGRESS
   ===================================================== */

[data-testid="stProgressBar"] {
    background: #e2e8f0 !important;
}


/* =====================================================
   FOOTER
   ===================================================== */

.footer {
    text-align: center;

    color: #cbd5e1 !important;

    margin-top: 40px;

    padding: 20px;

    font-size: 14px;
}

.footer * {
    color: #cbd5e1 !important;
}


/* =====================================================
   FILE NAME / UPLOADED IMAGE CAPTION
   ===================================================== */

[data-testid="stImage"] p {
    color: #cbd5e1 !important;
}


/* =====================================================
   REMOVE DEFAULT STREAMLIT TOP SPACE
   ===================================================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)




# =========================================================
# MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "convert_from_scratch_with_augmentation.keras"


@st.cache_resource
def load_model():

    return load_keras_model(MODEL_PATH)


try:

    model = load_model()

except Exception as e:

    st.error("❌ Model load nahi ho saka.")

    st.code(str(e))

    st.stop()


# =========================================================
# CLASS NAMES
# =========================================================
# NOTE:
# Tumhare notebook mein exactly ye order hai.

CLASS_NAMES = [
    "cat",
    "cow",
    "deer",
    "dog",
    "lion"
]


EMOJIS = {

    "cat": "🐱",

    "cow": "🐄",

    "deer": "🦌",

    "dog": "🐕",

    "lion": "🦁"

}


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🐾 Animal Classification CNN</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload an image and let the CNN identify the animal'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# COLUMNS
# =========================================================

left, right = st.columns(2)


# =========================================================
# UPLOAD
# =========================================================

with left:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📤 Upload Animal Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    image = None

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# PREDICTION
# =========================================================

with right:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🤖 AI Prediction")

    if image is None:

        st.info(
            "👆 Pehle image upload karein."
        )

    else:

        predict = st.button(
            "🔍 Predict Animal",
            use_container_width=True
        )

        if predict:

            with st.spinner(
                "CNN model image analyze kar raha hai..."
            ):

                try:

                    # =====================================
                    # SAME PREPROCESSING AS NOTEBOOK
                    # =====================================

                    img = image.resize(
                        (180, 180)
                    )

                    img_array = np.array(
                        img,
                        dtype=np.float32
                    )

                    # IMPORTANT:
                    # Notebook mein /255 nahi kiya gaya.
                    # Isliye yahan bhi normalization nahi karni.

                    # Add batch dimension
                    img_array = np.expand_dims(
                        img_array,
                        axis=0
                    )

                    # =====================================
                    # MODEL PREDICTION
                    # =====================================

                    pred = model.predict(
                        img_array
                    )

                    scores = pred[0]

                    # =====================================
                    # GET INDEX
                    # =====================================

                    predicted_index = int(
                        np.argmax(scores)
                    )

                    # =====================================
                    # GET CLASS
                    # =====================================

                    predicted_class = CLASS_NAMES[
                        predicted_index
                    ]

                    # =====================================
                    # CONFIDENCE
                    # =====================================

                    confidence = float(
                        np.max(scores) * 100
                    )

                    # =====================================
                    # RESULT
                    # =====================================

                    emoji = EMOJIS.get(
                        predicted_class,
                        "🐾"
                    )

                    st.markdown(
                        '<div class="result">',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'<div style="font-size:75px;">'
                        f'{emoji}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'<div class="animal">'
                        f'{predicted_class.capitalize()}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'<div class="confidence">'
                        f'Confidence: '
                        f'{confidence:.2f}%'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.progress(
                        min(
                            confidence / 100,
                            1.0
                        )
                    )

                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )

                    # =====================================
                    # ALL PREDICTIONS
                    # =====================================

                    st.write(
                        "### 📊 Prediction Probabilities"
                    )

                    for index, score in enumerate(
                        scores
                    ):

                        if index >= len(
                            CLASS_NAMES
                        ):
                            break

                        class_name = CLASS_NAMES[
                            index
                        ]

                        percentage = float(
                            score * 100
                        )

                        st.write(
                            f"**{class_name.capitalize()}** "
                            f"— {percentage:.2f}%"
                        )

                        st.progress(
                            min(
                                float(score),
                                1.0
                            )
                        )

                except Exception as e:

                    st.error(
                        "❌ Prediction ke waqt error aaya."
                    )

                    st.code(
                        str(e)
                    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    '🐾 Multi Animal Images Classification CNN'
    '</div>',
    unsafe_allow_html=True
)