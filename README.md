# 🧠 AI-Powered News & Research Summarizer

Get the latest news and research papers on any topic—summarized intelligently with AI. Powered by Streamlit, Hugging Face Transformers, Arxiv API, and Serper API.

🚀 **Demo**

Try it out on your local machine or deploy it with ease to [Streamlit Cloud](https://streamlit.io/cloud)!

✨ **Features**

* **🔍 Topic-based Search:** Enter any topic, and get real-time results.
* **📰 News Summarization:** Fetches and summarizes top news using the [Serper API](https://serper.dev) (Google Search API).
* **📚 Research Papers:** Fetches recent papers from [arXiv](https://arxiv.org/).
* **🤖 AI Summarization:** Uses Hugging Face's [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn) model to generate concise summaries.
* **⚡ Responsive UI:** Built with [Streamlit](https://streamlit.io/) for a clean and interactive experience.

🛠️ **Tech Stack**

* [Streamlit](https://streamlit.io/): Web interface
* [Hugging Face Transformers](https://huggingface.co/transformers/): Text summarization
* [ArXiv API](https://arxiv.org/help/api/): Scientific research data
* [Serper.dev](https://serper.dev/): Google Search results
* `requests`, `xml.etree.ElementTree`, `json`: API interaction

🖥️ **Installation**

```bash
# Clone the repository
git clone https://github.com/nkprajapati01/newssumirizer.git
cd ai-news-research-summarizer

# Install dependencies
pip install -r requirements.txt
```
🔐 **Setup Secrets**

Create a `.streamlit/secrets.toml` file with the following content:

```bash
# .streamlit/secrets.toml
SERPER_API_KEY = "your-serper-api-key-here"
```
🚦 Run the App
```bash
streamlit run app.py
```
👨‍💻 **Author**
* Neeraj Kumar Prajapati
* Made with ❤️ for curious minds and research enthusiasts.

🧠 **Future Ideas**
* **Add support for more APIs like Semantic Scholar or PubMed
* **Enable voice input for topic search
* **Add caching and performance optimizations
* **Deploy to Hugging Face Spaces or Streamlit Cloud

⭐️ **Show Your Support**
* If you found this project useful, please give it a ⭐️ and consider sharing it!
