import streamlit as st
import requests
import xml.etree.ElementTree as ET
import os
from datetime import datetime
from transformers import pipeline
import textwrap
import json

# Constants
SERPER_RESULT_COUNT = 5
ARXIV_RESULT_COUNT = 3
MAX_SUMMARY_TOKENS = 1000

# Load API keys
def get_api_keys():
    serper_key = st.secrets["SERPER_API_KEY"] if "SERPER_API_KEY" in st.secrets else os.environ.get("SERPER_API_KEY")

    if not serper_key:
        st.error("Serper API Key not found.")
    if not serper_key:
        return None
    return serper_key

# Search using Serper API
def search_serper(query, api_key, num_results=SERPER_RESULT_COUNT):
    st.info(f"Searching Serper for news: '{query}'")
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": num_results})
    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
        results = response.json()
        organic_results = [{
            'title': item.get('title', 'N/A'),
            'link': item.get('link', '#'),
            'snippet': item.get('snippet', 'N/A'),
            'source': item.get('source', item.get('displayLink', 'N/A')),
            'position': item.get('position')
        } for item in results.get('organic', [])]
        return organic_results
    except Exception as e:
        st.error(f"Serper API Error: {e}")
        return []

# Search arXiv for related papers
def search_arxiv(query, max_results=ARXIV_RESULT_COUNT):
    st.info(f"Searching arXiv for: '{query}'")
    base_url = 'https://export.arxiv.org/api/query?'
    search_query = f'search_query=all:"{query.replace(" ", "+")}"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}'
    url = base_url + search_query

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = []

        for entry in root.findall('atom:entry', namespace):
            arxiv_id = entry.find('atom:id', namespace).text.split('/abs/')[-1]
            title = entry.find('atom:title', namespace).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', namespace).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', namespace).text.split('T')[0]
            authors = [author.find('atom:name', namespace).text for author in entry.findall('atom:author', namespace)]
            entries.append({
                'title': title,
                'id': arxiv_id,
                'summary': summary,
                'published': published,
                'authors': authors,
                'url': f"https://arxiv.org/abs/{arxiv_id}"
            })
        return entries
    except Exception as e:
        st.error(f"arXiv API Error: {e}")
        return []

# Summarize everything using Hugging Face model
def summarize_with_ai(topic, serper_results, arxiv_results):
    st.info("Generating summary...")

    # Context for summarization
    context = f"Topic: {topic}\n\n--- Recent News ---\n"
    token_count = len(context.split())

    if serper_results:
        for item in serper_results:
            text = f"Title: {item['title']}\nSource: {item['source']}\nSnippet: {item['snippet']}\nLink: {item['link']}\n\n"
            if token_count + len(text.split()) < MAX_SUMMARY_TOKENS:
                context += text
                token_count += len(text.split())

    context += "\n--- Research Pre-prints ---\n"
    if arxiv_results:
        for item in arxiv_results:
            text = f"Title: {item['title']}\nAuthors: {', '.join(item['authors'])}\nPublished: {item['published']}\nSummary: {item['summary']}\nLink: {item['url']}\n\n"
            if token_count + len(text.split()) < MAX_SUMMARY_TOKENS:
                context += text
                token_count += len(text.split())

    # Ensure the context is within the token limit of the model (e.g., 1024 tokens)
    max_token_limit = 1024  # You can adjust this value based on your model's limit
    context_tokens = context.split()

    if len(context_tokens) > max_token_limit:
        context = " ".join(context_tokens[:max_token_limit])

    # Use Hugging Face transformer model for summarization
    summarizer = pipeline("summarization")
    try:
        summary = summarizer(context, max_length=500, min_length=50, do_sample=False)[0]['summary_text']
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        summary = "There was an error summarizing the content."

    return summary

# Show the results
def display_results(topic, summary, serper_data, arxiv_data):
    st.markdown(f"## Summary for: {topic}")
    st.subheader("AI Summary")
    st.write(textwrap.fill(summary, width=90))

    if st.checkbox("Show Raw Search Results"):
        st.subheader("Web Results")
        for i, item in enumerate(serper_data):
            st.markdown(f"**{i+1}. {item['title']}**\n\nSource: {item['source']}\n\n{item['snippet']}\n\n[Link]({item['link']})\n---")

        st.subheader("arXiv Results")
        for i, item in enumerate(arxiv_data):
            st.markdown(f"**{i+1}. {item['title']}**\n\nAuthors: {', '.join(item['authors'])}\nPublished: {item['published']}\n\n{item['summary']}\n\n[Link]({item['url']})\n---")

# Main Streamlit app
def main():
    st.title("🔍 Live News & Research Summarizer")
    serper_api_key = get_api_keys()

    with st.form("search_form"):
        user_topic = st.text_input("Enter a topic:")
        submitted = st.form_submit_button("Search and Summarize")

    if submitted:
        if not serper_api_key:
            st.error("API keys are missing or invalid.")
            return

        with st.spinner("Searching and summarizing..."):
            serper_results = search_serper(user_topic, serper_api_key)
            arxiv_results = search_arxiv(user_topic)
            summary = summarize_with_ai(user_topic, serper_results, arxiv_results)
            display_results(user_topic, summary, serper_results, arxiv_results)

if __name__ == "__main__":
    main()
