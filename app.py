import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the dataset
df = pd.read_csv("output_file.csv")

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment
def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    return "Positive" if score > 0 else "Negative"

# Apply modern styling using CSS
st.markdown(
    """
    <style>
        /* Background */
        .stApp {
            background-color: #1e3a5f; /* Dark bluish-gray */
            color: #ffffff; /* White text for contrast */
        }
        /* Title */
        .title {
            font-size: 30px;
            font-weight: bold;
            color: #f1c40f; /* Golden Yellow */
            text-align: center;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
        }
        /* Subheadings */
        .subheading {
            font-size: 22px;
            font-weight: bold;
            color: #f39c12; /* Warm orange */
        }
        /* Cards */
        .card {
            background-color: rgba(255, 255, 255, 0.15);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for selecting location
st.sidebar.title("📍 Select a Location")
locations = df["Location"].unique()
selected_location = st.sidebar.selectbox("Choose a place:", [""] + list(locations))  # Initially empty

if selected_location:
    # Filter data for the selected location
    filtered_data = df[df["Location"] == selected_location].copy()
    filtered_data["Sentiment"] = filtered_data["Review"].apply(get_sentiment)

    # Count positive and negative reviews
    positive_count = (filtered_data["Sentiment"] == "Positive").sum()
    negative_count = (filtered_data["Sentiment"] == "Negative").sum()
    total_reviews = positive_count + negative_count

    # Display sentiment counts
    st.markdown("<h2 class='title'>📊 Sentiment Analysis</h2>", unsafe_allow_html=True)
    st.subheader(f"📍 {selected_location}")

    st.markdown(f"<div class='card'><p class='subheading'>✅ Positive Reviews: <b>{positive_count}</b></p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card'><p class='subheading'>❌ Negative Reviews: <b>{negative_count}</b></p></div>", unsafe_allow_html=True)

    # Create a bar chart for sentiment visualization
    sentiment_data = pd.DataFrame({
        "Sentiment": ["Positive", "Negative"],
        "Count": [positive_count, negative_count]
    })

    fig = px.bar(
        sentiment_data,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},  # Green for positive, red for negative
        text="Count",
        title="📊 Sentiment Distribution",
        labels={"Count": "Number of Reviews", "Sentiment": "Review Type"}
    )

    fig.update_layout(
        plot_bgcolor="#1e3a5f",  # Match background
        paper_bgcolor="#1e3a5f",
        font=dict(color="white"),  # White text
        title_x=0.5  # Center title
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    # If no location is selected, allow users to analyze their own review
    st.markdown("<h2 class='title'>📝 Analyze Your Own Review</h2>", unsafe_allow_html=True)
    user_review = st.text_area("Enter your review:")
    if st.button("Predict Sentiment"):
        if user_review:
            sentiment = get_sentiment(user_review)
            st.markdown(f"<div class='card'><p class='subheading'>Sentiment: <b>{sentiment}</b></p></div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a review.")
