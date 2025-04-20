import streamlit as st
from transformers import pipeline
import arxiv
from serpapi.google_search import GoogleSearch  # CORRECTED IMPORT!

# Initialize Hugging Face summarizer
try:
    summarizer = pipeline("summarization")
except Exception as e:
    st.error(f"Error initializing summarization pipeline: {e}")
    summarizer = None

# Function to summarize text
def summarize_text(text):
    if summarizer:
        try:
            if len(text) > 1024:  # Token length limit for some models
                text = text[:1024]
            summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            st.error(f"Error during summarization: {e}")
            return ""
    else:
        return "Summarization pipeline not initialized."

# Function to fetch data from Arxiv
def fetch_arxiv_data(topic):
    try:
        search = arxiv.Search(query=topic, max_results=3) # Limiting results for faster processing
        results = ""
        for result in search.results():
            results += result.summary + " "
        return results
    except Exception as e:
        st.error(f"Error fetching data from Arxiv: {e}")
        return ""

# Function to fetch data from SerpAPI
def fetch_serper_data(topic):
    serpapi_key = st.secrets.get("SERPAPI_API_KEY")
    if serpapi_key:
        try:
            client = GoogleSearch({"q": topic, "api_key": serpapi_key})
            results = client.get_dict()
            organic_results = results.get("organic_results", [])
            search_snippets = " ".join([result.get("snippet", "") for result in organic_results])
            return search_snippets
        except Exception as e:
            st.error(f"Error fetching data from SerpAPI: {e}")
            return ""
    else:
        st.warning("Please enter your SerpAPI API key in Streamlit Secrets.")
        return ""

# Streamlit UI
def main():
    st.title("AI-powered News & Research Summarizer")

    # Input topic or query
    user_topic = st.text_input("Enter the topic you want to summarize:")

    if user_topic:
        with st.spinner("Fetching and summarizing data..."):
            arxiv_results = fetch_arxiv_data(user_topic)
            serper_results = fetch_serper_data(user_topic)

            context = arxiv_results + " " + serper_results
            summary = summarize_text(context)

            if summary:
                st.subheader("Summary:")
                st.write(summary)
            else:
                st.info("Could not generate a summary. Please check for any error messages.")

if __name__ == "__main__":
    main()
