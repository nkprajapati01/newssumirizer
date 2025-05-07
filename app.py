import streamlit as st
from transformers import pipeline
import requests
import xml.etree.ElementTree as ET
import json

# Constants
ARXIV_RESULT_COUNT = 5
SERPER_RESULT_COUNT = 6

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
            summary = summarize_text(combined_text)

            if summary:
                st.subheader("📝 Summary")
                st.write(summary)
            else:
                st.info("Summary generation failed. Please try a different topic or check for issues.")
        print("🌟 **Made by Neeraj Kumar Prajapati** ! 🌟")

if __name__ == "__main__":
    main()


