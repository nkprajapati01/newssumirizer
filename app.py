import streamlit as st
from transformers import pipeline
import arxiv
import serpapi

# Initialize Hugging Face summarizer
summarizer = pipeline("summarization")

def summarize_text(text):
    if len(text) > 1024:  # Token length limit
        text = text[:1024]
    summary = summarizer(text)
    return summary[0]['summary_text']

# Streamlit UI
def main():
    st.title("AI-powered News Summarizer")
    
    # Input topic or query
    user_topic = st.text_input("Enter the topic you want to summarize:")
    
    if user_topic:
        # Fetch data from Arxiv API (or your preferred source)
        arxiv_results = fetch_arxiv_data(user_topic)  # This is a placeholder for your Arxiv query
        serper_results = fetch_serper_data(user_topic)  # This is a placeholder for your SerpAPI data

        context = arxiv_results + " " + serper_results
        summary = summarize_text(context)
        
        st.write("Summary: ", summary)

def fetch_arxiv_data(topic):
    # Example function to fetch data from arxiv
    search = arxiv.Search(query=topic, max_results=5)
    results = ""
    for result in search.results():
        results += result.summary + " "
    return results

def fetch_serper_data(topic):
    # Example function to fetch data from SerpAPI
    client = serpapi.GoogleSearch({"q": topic, "api_key": "YOUR_SERPAPI_KEY"})
    results = client.get_dict()
    return results.get("organic_results", "")

if __name__ == "__main__":
    main()
