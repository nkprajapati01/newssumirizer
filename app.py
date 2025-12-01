import streamlit as st
from transformers import pipeline
import requests
import xml.etree.ElementTree as ET
import json

# Constants
ARXIV_RESULT_COUNT = 5
SERPER_RESULT_COUNT = 6

# Available summarization models to pick from
MODEL_OPTIONS = [
    "facebook/bart-large-cnn",
    # "t5-base",
    # "t5-small",
    # "google/pegasus-xsum",
    # "sshleifer/distilbart-cnn-12-6",
    # "declare-lab/flan-t5-base"
]

# Initialize Hugging Face summarizer (cached per model_name)
@st.cache_resource(show_spinner=False)
def load_summarizer(model_name: str):
    try:
        # If you have a GPU available and want to use it, set device=0.
        # For CPU usage, pipeline will run on CPU by default.
        # To automatic GPU detection, you can uncomment and use torch if installed.
        # import torch
        # device = 0 if torch.cuda.is_available() else -1
        # return pipeline("summarization", model=model_name, device=device)
        return pipeline("summarization", model=model_name)
    except Exception as e:
        st.error(f"Error initializing summarization pipeline for {model_name}: {e}")
        return None

# Function to summarize text (accepts the pipeline instance)
def summarize_text(text, summarizer):
    if not summarizer:
        return "Summarization pipeline not initialized."
    try:
        if len(text) > 1024:  # Truncate long input; adjust for your chosen model's limits
            text = text[:1024]
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return ""

# Function to fetch data from Serper API
def fetch_serper_data(topic):
    serper_api_key = st.secrets.get("SERPER_API_KEY")
    if not serper_api_key:
        st.warning("Please add your Serper API key to Streamlit secrets.")
        return []

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": topic, "num": SERPER_RESULT_COUNT})
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
        results = response.json()
        organic_results = results.get("organic", [])
        return [{
            'title': item.get('title', 'N/A'),
            'snippet': item.get('snippet', 'N/A'),
            'source': item.get('source', 'N/A'),
            'url': item.get('link', '#')
        } for item in organic_results]
    except Exception as e:
        st.error(f"Error fetching data from Serper API: {e}")
        return []

# Function to fetch data from Arxiv
def search_arxiv(query, max_results=ARXIV_RESULT_COUNT):
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'search_query=all:"{query.replace(" ", "+")}"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}'
    url = base_url + search_query

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = []

        for entry in root.findall('atom:entry', namespace):
            arxiv_id_raw = entry.find('atom:id', namespace).text
            arxiv_id = arxiv_id_raw.split('/abs/')[-1]
            title = entry.find('atom:title', namespace).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', namespace).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', namespace).text
            authors = [a.find('atom:name', namespace).text for a in entry.findall('atom:author', namespace)]

            entries.append({
                'title': title,
                'id': arxiv_id,
                'summary': summary,
                'published': published.split('T')[0],
                'authors': authors,
                'url': f"https://arxiv.org/abs/{arxiv_id}"
            })

        return entries

    except Exception as e:
        st.error(f"Error fetching data from Arxiv: {e}")
        return []

# Streamlit UI
def main():
    st.title("🧠 AI-powered News & Research Summarizer")

    # Let user choose the model
    selected_model = st.selectbox("Choose summarization model:", MODEL_OPTIONS, index=MODEL_OPTIONS.index("facebook/bart-large-cnn"))

    # Load summarizer for selected model (cached by model_name)
    summarizer = load_summarizer(selected_model)

    user_topic = st.text_input("🔍 Enter a topic to summarize:")

    if user_topic:
        with st.spinner("Fetching information and generating summary..."):
            serper_results = fetch_serper_data(user_topic)
            arxiv_results = search_arxiv(user_topic)

            # Display News Results
            if serper_results:
                st.subheader("📰 Latest News")
                for result in serper_results:
                    st.markdown(f"**{result['title']}**")
                    st.write(f"{result['snippet']}")
                    st.write(f"Source: {result['source']}")
                    st.write(f"[Read more...]({result['url']})\n")

            # Display Research Papers
            if arxiv_results:
                st.subheader("📚 Latest Research Papers")
                for paper in arxiv_results:
                    st.markdown(f"**{paper['title']}**")
                    st.write(f"Authors: {', '.join(paper['authors'])}")
                    st.write(f"Published: {paper['published']}")
                    st.write(f"Summary: {paper['summary']}")
                    st.write(f"[Read paper...]({paper['url']})\n")

            # Combine all results for summarization
            combined_text = " ".join([result['snippet'] for result in serper_results]) + " " + \
                            " ".join([paper['summary'] for paper in arxiv_results])
            summary = summarize_text(combined_text, summarizer)

            if summary:
                st.subheader("📝 Summary")
                st.write(summary)
            else:
                st.info("Summary generation failed. Please try a different topic or check for issues.")
    print("🌟 **Made by Neeraj Kumar Prajapati** ! 🌟")

if __name__ == "__main__":
    main()
