import streamlit as st
from transformers import pipeline, AutoTokenizer # Import tokenizer for better length check
import arxiv
import serpapi
import logging # Import logging for better error tracking

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Consider using a specific model for potentially better/faster results
# Examples: 'sshleifer/distilbart-cnn-6-6' (smaller), 'facebook/bart-large-cnn' (often default)
MODEL_NAME = "facebook/bart-large-cnn"
# Using AutoTokenizer to match the pipeline's tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Get the actual model max length, subtract some buffer for safety/special tokens
    # Common limits are 512 or 1024 for summarization models like BART
    MAX_TOKENS = tokenizer.model_max_length - 20 # Default to model max length with buffer
    logging.info(f"Using tokenizer '{MODEL_NAME}' with max token limit (approx): {MAX_TOKENS}")
except Exception as e:
    logging.warning(f"Could not load tokenizer for {MODEL_NAME}. Falling back to character limit. Error: {e}")
    MAX_TOKENS = 1000 # Fallback if tokenizer fails
    tokenizer = None # Ensure tokenizer is None if failed

# Character limit (fallback or alternative, less accurate than tokens)
MAX_CHARS_INPUT = 8000 # Increased fallback limit

# --- Hugging Face Initialization ---
# Use st.cache_data to cache the pipeline object, preventing re-downloading the model on every run
@st.cache_resource # Use cache_resource for non-data objects like models
def load_summarizer():
    logging.info(f"Loading summarization pipeline with model: {MODEL_NAME}")
    try:
        summarizer = pipeline("summarization", model=MODEL_NAME, device=-1) # device=-1 ensures CPU usage if no GPU
        logging.info("Summarization pipeline loaded successfully.")
        return summarizer
    except Exception as e:
        st.error(f"Failed to load the summarization model ({MODEL_NAME}): {e}")
        logging.error(f"Failed to load summarization model: {e}", exc_info=True)
        return None

summarizer = load_summarizer()

# --- Data Fetching Functions ---

def fetch_arxiv_data(topic, max_results=3):
    """Fetches summaries from ArXiv based on the topic."""
    logging.info(f"Fetching ArXiv data for topic: {topic}")
    results_text = ""
    try:
        search = arxiv.Search(query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        results = list(search.results()) # Consume the generator
        if not results:
            logging.info("No results found on ArXiv.")
            return ""
        for result in results:
            results_text += f"ArXiv Paper: {result.title}\nSummary: {result.summary}\n\n"
        logging.info(f"Fetched {len(results)} results from ArXiv.")
        return results_text.strip()
    except Exception as e:
        st.error(f"Failed to fetch data from ArXiv: {e}")
        logging.error(f"ArXiv fetch error: {e}", exc_info=True)
        return ""

def fetch_serper_data(topic, max_results=5):
    """Fetches organic search result snippets from Google via SerpApi."""
    logging.info(f"Fetching SerpApi data for topic: {topic}")
    # --- Use Streamlit Secrets for API Key ---
    if "SERPAPI_KEY" not in st.secrets:
        st.error("SerpApi API key not found. Please add it to your Streamlit secrets.")
        logging.error("SerpApi API key missing in st.secrets.")
        return ""
    api_key = st.secrets["SERPAPI_KEY"]
    # --- End Secret Handling ---

    results_text = ""
    try:
        params = {
            "q": topic,
            "api_key": api_key,
            "num": max_results  # Control number of results
        }
        client = serpapi.GoogleSearch(params)
        results_dict = client.get_dict()

        organic_results = results_dict.get("organic_results", [])
        if not organic_results:
            logging.info("No organic results found via SerpApi.")
            return ""

        snippets = []
        for result in organic_results:
            snippet = result.get("snippet", "")
            title = result.get("title", "Untitled")
            link = result.get("link", "#")
            if snippet:
                snippets.append(f"Web Result: {title}\nSnippet: {snippet}\nLink: {link}\n\n")

        results_text = "".join(snippets)
        logging.info(f"Fetched {len(snippets)} snippets via SerpApi.")
        return results_text.strip()
    except Exception as e:
        st.error(f"Failed to fetch data from SerpApi: {e}")
        logging.error(f"SerpApi fetch error: {e}", exc_info=True)
        return ""

# --- Summarization Function ---

def summarize_text(text, min_length=50, max_length=200):
    """Summarizes the input text using the Hugging Face pipeline."""
    if not summarizer:
        st.error("Summarizer model not loaded. Cannot summarize.")
        return "Error: Summarizer not available."
    if not text or not text.strip():
        logging.warning("Attempted to summarize empty text.")
        return "No text provided to summarize."

    logging.info(f"Summarizing text. Original length: {len(text)} chars.")

    # --- Token-based length check (preferred) ---
    if tokenizer:
        try:
            tokens = tokenizer(text, return_tensors="pt", truncation=False)['input_ids']
            num_tokens = tokens.shape[1]
            logging.info(f"Number of tokens: {num_tokens}")
            if num_tokens > MAX_TOKENS:
                st.warning(f"Input text ({num_tokens} tokens) exceeds model limit ({MAX_TOKENS} tokens). Truncating.")
                logging.warning(f"Input text ({num_tokens} tokens) exceeds model limit ({MAX_TOKENS} tokens). Truncating.")
                # Truncate based on tokens (more accurate)
                truncated_ids = tokens[0, :MAX_TOKENS].unsqueeze(0)
                text = tokenizer.decode(truncated_ids[0], skip_special_tokens=True)
                logging.info(f"Truncated text length: {len(text)} chars.")
        except Exception as e:
            st.error(f"Error during tokenization or truncation: {e}")
            logging.error(f"Tokenization/Truncation error: {e}", exc_info=True)
            # Fallback to character truncation if tokenization fails
            if len(text) > MAX_CHARS_INPUT:
                logging.warning(f"Falling back to character truncation. Original length: {len(text)} chars.")
                text = text[:MAX_CHARS_INPUT]
                logging.info(f"Truncated text length: {len(text)} chars.")

    # --- Fallback Character-based length check ---
    elif len(text) > MAX_CHARS_INPUT:
        st.warning(f"Input text ({len(text)} chars) exceeds limit ({MAX_CHARS_INPUT} chars). Truncating. Consider using a model with a tokenizer for better accuracy.")
        logging.warning(f"Input text ({len(text)} chars) exceeds limit ({MAX_CHARS_INPUT} chars). Truncating.")
        text = text[:MAX_CHARS_INPUT]
        logging.info(f"Truncated text length: {len(text)} chars.")

    try:
        logging.info("Calling summarizer pipeline...")
        # Adjust min/max length as needed
        summary_list = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        if summary_list and isinstance(summary_list, list) and 'summary_text' in summary_list[0]:
            summary = summary_list[0]['summary_text']
            logging.info(f"Summary generated successfully. Length: {len(summary)} chars.")
            return summary
        else:
            logging.error(f"Summarizer returned unexpected output: {summary_list}")
            return "Error: Could not generate summary from the model output."
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        logging.error(f"Summarization error: {e}", exc_info=True)
        return f"Error during summarization process: {e}"


# --- Streamlit UI ---
def main():
    st.set_page_config(
        page_title="AI News + Research Summarizer",
        page_icon="🧠",
        layout="wide" # Use wide layout as suggested in config
    )
    st.title("AI-powered News + Research Summarizer")
    st.markdown("Enter a topic to fetch recent ArXiv papers and Google search results, then summarize them.")

    # Input topic or query
    user_topic = st.text_input("Enter the topic you want to summarize:", key="topic_input")

    if user_topic:
        # Use columns for better layout
        col1, col2 = st.columns(2)

        arxiv_results = ""
        serper_results = ""

        with col1:
            st.subheader("Fetching ArXiv Data...")
            with st.spinner('Searching ArXiv...'):
                arxiv_results = fetch_arxiv_data(user_topic)
            if arxiv_results:
                with st.expander("View ArXiv Results", expanded=False):
                    st.text_area("ArXiv Content", arxiv_results, height=200, key="arxiv_results_text")
            else:
                st.info("No relevant papers found on ArXiv for this topic.")

        with col2:
            st.subheader("Fetching Web Data (via SerpApi)...")
            with st.spinner('Searching Google via SerpApi...'):
                serper_results = fetch_serper_data(user_topic)
            if serper_results:
                 with st.expander("View Web Search Results", expanded=False):
                    st.text_area("Web Content", serper_results, height=200, key="serper_results_text")
            else:
                st.info("No relevant web results found via SerpApi for this topic.")

        # Combine results
        context = ""
        if arxiv_results:
            context += "--- ArXiv Results ---\n" + arxiv_results + "\n\n"
        if serper_results:
            context += "--- Web Search Results ---\n" + serper_results

        if context.strip() and summarizer: # Check if context exists and summarizer loaded
            st.subheader("Combined Summary")
            with st.spinner('Generating summary... This may take a moment.'):
                summary = summarize_text(context.strip()) # Use strip()
                st.success("Summary generated!") # Use success box for summary
                st.markdown(summary) # Use markdown for potentially better formatting
        elif not context.strip():
            st.warning("No content fetched from ArXiv or Web Search to summarize.")
        elif not summarizer:
             st.error("Summary could not be generated because the summarization model failed to load.")


if __name__ == "__main__":
    main()
