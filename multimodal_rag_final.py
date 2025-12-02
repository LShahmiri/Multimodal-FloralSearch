import streamlit as st
from datasets import load_dataset
import os
from PIL import Image
import base64
import warnings
from dotenv import load_dotenv

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


warnings.filterwarnings("ignore")
load_dotenv()

# ------------------------------------------------------------
# UI: Modern Pink/Gold Dashboard (No AI Powered Text)
# ------------------------------------------------------------
st.set_page_config(page_title="Flower Designer", page_icon="🌸", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background: #fff6fb;
}

.header-box {
    background: linear-gradient(135deg, #ff78b9, #ffb27b);
    padding: 40px;
    text-align: center;
    border-radius: 20px;
    margin-bottom: 25px;
    color: white;
    box-shadow: 0 8px 25px rgba(0,0,0,0.18);
}

.header-title {
    font-size: 46px;
    font-weight: 800;
}

.search-box {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    margin-bottom: 25px;
}

.search-btn {
    background: linear-gradient(135deg, #ff78b9, #ffb27b);
    color: white !important;
    border: none;
    padding: 12px 28px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 600;
    width: 100%;
    cursor: pointer;
}

.search-btn:hover {
    background: linear-gradient(135deg, #ff629f, #ffa565);
}

.suggest-box {
    background: #fff1fa;
    padding: 30px;
    border-radius: 16px;
    border-left: 6px solid #d81b60;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-box">
    <div class="header-title">🌸 Flower Arrangement Query & Image Retrieval</div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
@st.cache_data
def load_flower_dataset():
    ds = load_dataset("huggan/flowers-102-categories")

    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)

    paths = []
    for i, img in enumerate(ds["train"]["image"][:200]):   # limit for speed
        path = f"{image_dir}/flower_{i}.jpg"
        img.save(path)
        paths.append(path)

    return paths

image_paths = load_flower_dataset()

# ------------------------------------------------------------
# Setup ChromaDB
# ------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path="./data/flower.db")

image_loader = ImageLoader()
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection(
    name="flowers",
    embedding_function=embedding_fn,
    data_loader=image_loader
)

if collection.count() == 0:
    collection.add(
        ids=[str(i) for i in range(len(image_paths))],
        uris=image_paths
    )

# ------------------------------------------------------------
# Helper: Display Image
# ------------------------------------------------------------
def show_image(uri):
    st.image(Image.open(uri), width=240)

# ------------------------------------------------------------
# AI Query Validator (FIXED VERSION)
# ------------------------------------------------------------
def ai_validate_query(query):

    validator = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
    You are a classifier. Determine if the user query is describing a flower visually.
    Respond ONLY with YES or NO.

    Query: "{query}"

    Criteria for YES:
    - Mentions colors (pink, yellow, purple…)
    - Mentions flower parts (petals, center, leaves…)
    - Mentions flower names (rose, orchid, daisy…)
    - Mentions shape or texture of a flower

    Respond YES if it is about a flower description.
    Otherwise respond NO.
    """

    response = validator.invoke(prompt)
    result = response.content.strip().upper()     # FIXED

    return result == "YES"

# ------------------------------------------------------------
# Format LLM Prompt
# ------------------------------------------------------------
def format_prompt_inputs(data, user_query):

    img1 = data["uris"][0][0]
    img2 = data["uris"][0][1]

    def encode(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    return {
        "user_query": user_query,
        "image_data_1": encode(img1),
        "image_data_2": encode(img2),
    }

# ------------------------------------------------------------
# Query ChromaDB
# ------------------------------------------------------------
def query_db(q):
    return collection.query(
        query_texts=[q],
        n_results=2,
        include=["uris"]
    )

# ------------------------------------------------------------
# LLM
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return ChatOpenAI(model="gpt-4o", temperature=0.0)

model = load_model()
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional florist. Provide bouquet recommendations based on the images."),
    ("user", [
        {"type": "text", "text": "{user_query}"},
        {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_1}"},
        {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_2}"},
    ])
])

vision_chain = prompt | model | parser

# ------------------------------------------------------------
# Search Box UI
# ------------------------------------------------------------
st.markdown("<div class='search-box'>", unsafe_allow_html=True)

query = st.text_input(
    "Describe your flower search:",
    placeholder="e.g., elegant pink lilies for birthday",
    label_visibility="collapsed"
)

search_clicked = st.button("Search", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# SEARCH + VALIDATION
# ------------------------------
if search_clicked:

    if not query.strip():
        st.error("⚠ Please enter a description.")
        st.stop()

    # Validate input
    if not ai_validate_query(query):
        st.error("❌ This query does not describe a flower. Please describe a flower's appearance.")
        st.stop()

    # Search successfully
    st.write(f"🔍 Searching for: **{query}**")

    with st.spinner("Retrieving best matches..."):
        results = query_db(query)

    st.subheader(" Matched Flowers")
    cols = st.columns(2)
    for i, img_path in enumerate(results["uris"][0]):
        cols[i % 2].image(img_path, width=250)

    with st.spinner("Generating bouquet ideas..."):
        data = format_prompt_inputs(results, query)
        output = vision_chain.invoke(data)

    st.markdown("<div class='suggest-box'>", unsafe_allow_html=True)
    st.subheader(" Bouquet Arrangement Suggestions")
    st.write(output)
    st.markdown("</div>", unsafe_allow_html=True)
