🧠 AI-Powered News & Research Summarizer
Get the latest news and research papers on any topic—summarized intelligently with AI. Powered by Streamlit, Hugging Face Transformers, Arxiv API, and Serper API.

🚀 Demo
Try it out on your local machine or deploy it with ease to Streamlit Cloud!

✨ Features
🔍 Topic-based Search: Enter any topic, and get real-time results.

📰 News Summarization: Fetches and summarizes top news using the Serper API (Google Search API).

📚 Research Papers: Fetches recent papers from arXiv.

🤖 AI Summarization: Uses Hugging Face's facebook/bart-large-cnn model to generate concise summaries.

⚡ Responsive UI: Built with Streamlit for a clean and interactive experience.

🛠️ Tech Stack
Streamlit for web interface

Hugging Face Transformers for text summarization

ArXiv API for scientific research data

Serper.dev for Google Search results

requests, xml.etree.ElementTree, json for API interaction

🖥️ Installation

# Clone the repository
git clone https://github.com/nkprajapati01/newssumirizer
cd ai-news-research-summarizer

# Install dependencies
pip install -r requirements.txt
Note: You will need a Serper API key. Get it at serper.dev.

🔐 Setup Secrets
Create a .streamlit/secrets.toml file with the following content:


SERPER_API_KEY = "your-serper-api-key-here"
🚦 Run the App

streamlit run app.py
📸 Screenshot
Add a screenshot of the app here to visually showcase it on GitHub.

👨‍💻 Author
Neeraj Kumar Prajapati

Made with ❤️ for curious minds and research enthusiasts.

🧠 Future Ideas
Add support for more APIs like Semantic Scholar or PubMed

Enable voice input for topic search

Add caching and performance optimizations

Deploy to Hugging Face Spaces or Streamlit Cloud

⭐️ Show Your Support
If you found this project useful, please give it a ⭐️ and consider sharing it!

