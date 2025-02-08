import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the dataset
df = pd.read_csv("output_file.csv")

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment
def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    return "Positive" if score > 0 else "Negative"

# Apply custom CSS for a colorful background
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #ff9a9e, #fad0c4);
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #fff;
            text-align: center;
            padding: 10px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
        }
        .positive {
            color: #28a745;
            font-weight: bold;
            font-size: 20px;
        }
        .negative {
            color: #dc3545;
            font-weight: bold;
            font-size: 20px;
        }
        .sidebar-title {
            font-size: 22px;
            font-weight: bold;
            color: #fff;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for selecting location
st.sidebar.markdown("<p class='sidebar-title'>📍 Select a Location</p>", unsafe_allow_html=True)
locations = df["Location"].unique()
selected_location = st.sidebar.selectbox("Choose a place:", [""] + list(locations))  # Default as empty

if selected_location:
    # Filter data for the selected location
    filtered_data = df[df["Location"] == selected_location].copy()
    filtered_data["Sentiment"] = filtered_data["Review"].apply(get_sentiment)

    # Count positive and negative reviews
    positive_count = (filtered_data["Sentiment"] == "Positive").sum()
    negative_count = (filtered_data["Sentiment"] == "Negative").sum()

    # Display sentiment counts
    st.markdown("<h2 class='title'>📊 Sentiment Analysis</h2>", unsafe_allow_html=True)
    st.subheader(f"📍 {selected_location}")
    st.write(f"✅ <span class='positive'>Positive Reviews: {positive_count}</span>", unsafe_allow_html=True)
    st.write(f"❌ <span class='negative'>Negative Reviews: {negative_count}</span>", unsafe_allow_html=True)

# User review input
st.markdown("<h2 class='title'>✍️ Analyze Your Own Review</h2>", unsafe_allow_html=True)
user_review = st.text_area("Enter your review:")
if st.button("Predict Sentiment"):
    if user_review:
        sentiment = get_sentiment(user_review)
        color_class = "positive" if sentiment == "Positive" else "negative"
        st.markdown(f"**Sentiment:** <span class='{color_class}'>{sentiment}</span>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a review.")
