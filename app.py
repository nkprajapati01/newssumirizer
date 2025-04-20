import streamlit as st
from transformers import pipeline
import arxiv
import requests
import json

# Initialize Hugging Face summarizer
@st.cache_resource(show_spinner=False)
def load_summarizer():
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn")  # Using an optimal open-source model
    except Exception as e:
        st.error(f"Error initializing summarization pipeline: {e}")
        return None

summarizer = load_summarizer()

# Function to summarize text
def summarize_text(text):
    if summarizer:
        try:
            if len(text) > 1024:  # Truncate long input
                text = text[:1024]
            summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            st.error(f"Error during summarization: {e}")
            return ""
    return "Summarization pipeline not initialized."

# Function to fetch data from Arxiv
def fetch_arxiv_data(topic):
    try:
        search = arxiv.Search(query=topic, max_results=3)
        results = ""
        for result in search.results():
            results += result.summary + " "
        return results.strip()
    except Exception as e:
        st.error(f"Error fetching data from Arxiv: {e}")
        return ""

# Function to fetch data from Serper API
def fetch_serper_data(topic):
    serper_api_key = st.secrets.get("SERPER_API_KEY")
    if not serper_api_key:
        st.warning("Please add your Serper API key to Streamlit secrets.")
        return ""

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": topic})
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
        results = response.json()
        organic_results = results.get("organic", [])
        snippets = [item.get("snippet", "") for item in organic_results if item.get("snippet")]
        return " ".join(snippets)
    except Exception as e:
        st.error(f"Error fetching data from Serper API: {e}")
        return ""

# Streamlit UI
def main():
    st.title("🧠 AI-powered News & Research Summarizer")

    user_topic = st.text_input("🔍 Enter a topic to summarize:")

    if user_topic:
        with st.spinner("Fetching information and generating summary..."):
            arxiv_results = fetch_arxiv_data(user_topic)
            serper_results = fetch_serper_data(user_topic)

            combined_text = arxiv_results + " " + serper_results
            if not combined_text.strip():
                st.warning("No relevant information was found.")
                return

            summary = summarize_text(combined_text)

            if summary:
                st.subheader("📝 Summary")
                st.write(summary)
            else:
                st.info("Summary generation failed. Please try a different topic or check for issues.")

if __name__ == "__main__":
    main()
