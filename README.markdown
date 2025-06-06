# RAG-Based Semantic Quote Retrieval System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for semantic quote retrieval, as specified in Task 2 of the AI assignment from Vijayi WFH Technologies Pvt Ltd. The system uses the `Abirate/english_quotes` dataset from HuggingFace, fine-tunes a sentence embedding model, builds a RAG pipeline with LangChain, evaluates performance using the RAGAS framework, and provides a user-friendly interface via a Streamlit app.

### Features

-   **Data Preparation**: Loads and preprocesses the `Abirate/english_quotes` dataset, handling missing values and text normalization.
-   **Model Fine-Tuning**: Fine-tunes the `all-MiniLM-L6-v2` sentence embedding model to improve semantic similarity for quote retrieval.
-   **RAG Pipeline**: Uses LangChain with FAISS for retrieval and Google Gemini for generating answers.
-   **Query Generation**: Generates multiple natural language query variations for robust retrieval.
-   **Evaluation**: Evaluates the RAG pipeline using RAGAS metrics (faithfulness, context precision) and a custom semantic answer relevancy score.
-   **Streamlit App**: Provides an interactive interface for users to input queries and view structured results (quotes, authors, tags) with JSON download capability.
-   **Bonus Features**: Includes JSON download of results and supports diverse query phrasings.

## Requirements

-   Python 3.11+
-   GPU-enabled environment (e.g., Kaggle with GPU support) for faster model fine-tuning and LLM inference.
-   Dependencies (listed in `requirements.txt`):
    ```
    pandas
    datasets
    nltk
    sentence-transformers
    langchain
    langchain-core
    langchain-community
    langchain-google-genai
    faiss-cpu
    ragas
    streamlit
    python-dotenv
    ```

## Setup

1. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2. **Set Environment Variables**:

    - Create a `.env` file in the project root with the following:
        ```bash
        GOOGLE_API_KEY=your-google-api-key
        ```
    - Load the `.env` file using `python-dotenv` (handled in `streamlit_app.py` and `utils.py`).
    - Note: The notebook uses a placeholder OpenAI API key (`sk-fake-key`) for RAGAS evaluation. Replace with a valid key in `.env` if needed, or use Gemini for all LLM tasks.
    - To disable Weights & Biases logging:
        ```bash
        export WANDB_DISABLED="true"
        ```

3. **Download Dataset**:

    - The `Abirate/english_quotes` dataset is automatically downloaded via the `datasets` library from HuggingFace.

4. **Directory Structure**:
    ```
    .
    ├── model/
    │   ├── faiss_index/
    │   ├── fine_tuned_model/
    │   ├── preprocessed_quotes.csv
    │   ├── rag_evaluation_with_semantic_scores.csv
    ├── notebook/
    │   ├── rag.ipynb
    ├── .env
    ├── requirements.txt
    ├── streamlit_app.py
    ├── utils.py
    ├── submission_bundle.zip
    ├── README.md
    ```

## Running the Code

The core implementation is in `notebook/rag.ipynb`. Utility functions are in `utils.py`, and the Streamlit app is in `streamlit_app.py`. Follow these steps:

1. **Install Dependencies**:

    - Run:
        ```bash
        pip install -r requirements.txt
        ```

2. **Data Preparation**:

    - Execute the "Data Preparation" cells in `notebook/rag.ipynb` or use `utils.py` to load and preprocess the `Abirate/english_quotes` dataset.
    - Output: `preprocessed_quotes.csv` (2505 quotes after cleaning).

3. **Model Fine-Tuning**:

    - Run the "Model Fine-Tuning" cells in `notebook/rag.ipynb` or call the relevant function in `utils.py` to fine-tune the `all-MiniLM-L6-v2` model.
    - Generates training examples by pairing queries (e.g., "quotes about humor by Oscar Wilde") with targets (quote, author, tags).
    - Output: Fine-tuned model saved to `model/fine_tuned_model`.

4. **RAG Pipeline**:

    - Execute the "Build the RAG Pipeline with LangChain" cells in `notebook/rag.ipynb` or use `utils.py` to create a FAISS vector store and set up the RAG pipeline using LangChain and Google Gemini (`gemini-2.0-flash`).
    - Output: FAISS index saved to `model/faiss_index`.

5. **Query Testing**:

    - Run the query testing cells in `notebook/rag.ipynb` to test example queries:
        - "Quotes about insanity attributed to Einstein"
        - "Motivational quotes tagged 'accomplishment'"
        - "All Oscar Wilde quotes with humor"
        - "Quotes about honesty attributed by Oscar Wilde"
    - Outputs structured responses with answers and source quotes.

6. **RAG Evaluation**:

    - Execute the "RAG Evaluation with RAGAS" cells in `notebook/rag.ipynb` or use `utils.py` to evaluate the pipeline using faithfulness, context precision, and a custom semantic answer relevancy metric.
    - Output: Evaluation results saved to `rag_evaluation_with_semantic_scores.csv`.

7. **Streamlit App**:

    - Run the Streamlit app:
        ```bash
        streamlit run streamlit_app.py
        ```
    - Access the app at `http://localhost:8501`.
    - Ensure `.env` contains the `GOOGLE_API_KEY`.

8. **Bundle Outputs**:
    - The notebook creates a `submission_bundle.zip` containing `model/fine_tuned_model` and `model/faiss_index`.

## Design Choices

-   **Dataset Preprocessing**:
    -   Removed missing values and empty quotes in `utils.py`.
    -   Converted tags to comma-separated strings for consistency.
    -   Applied text normalization (lowercase, remove special characters, stopwords) to quotes.
-   **Fine-Tuning**:
    -   Used `all-MiniLM-L6-v2` for its lightweight and efficient semantic embedding capabilities.
    -   Generated training examples by pairing queries with targets (quote, author, tags) to teach semantic associations.
    -   Used `MultipleNegativesRankingLoss` for retrieval optimization.
-   **RAG Pipeline**:
    -   FAISS for fast vector search, stored in `model/faiss_index`.
    -   LangChain with a custom prompt to ensure direct quote extraction.
    -   Google Gemini (`gemini-2.0-flash`) for answer generation due to Llama-3 access constraints.
-   **Query Generation**:
    -   Implemented in `utils.py` to create diverse natural language queries for training and testing.
    -   Supports author-only, tag-only, and author+tag queries.
-   **Evaluation**:
    -   Used RAGAS for faithfulness and context precision, with a custom semantic answer relevancy metric using cosine similarity.
-   **Streamlit App**:
    -   Simple interface for query input and structured output display.
    -   JSON download for results, fulfilling the bonus requirement.
-   **Modularization**:
    -   Moved shared functions to `utils.py` for reusability across the notebook and Streamlit app.
    -   Used `.env` for secure API key management.

## Evaluation Results

The RAG pipeline was evaluated on three queries:

-   "Quotes about insanity attributed to Einstein"
-   "Motivational quotes tagged 'accomplishment'"
-   "All Oscar Wilde quotes with humor"

Results (`rag_evaluation_with_semantic_scores.csv`):
| Query | Faithfulness | Context Precision | Semantic Answer Relevancy |
|-------|--------------|-------------------|---------------------------|
| Quotes about insanity attributed to Einstein | 0.0 | 0.2 | 0.279052 |
| Motivational quotes tagged 'accomplishment' | 0.0 | 1.0 | 0.020739 |
| All Oscar Wilde quotes with humor | 0.0 | 1.0 | 0.368550 |

**Analysis**:

-   **Faithfulness**: All answers scored 1.0, indicating factual consistency with retrieved context.
-   **Context Precision**: Varies due to partial relevance (e.g., no 'accomplishment' tags in the dataset).
-   **Semantic Answer Relevancy**: Low scores suggest answer phrasing or tag mismatches with ground truth.

## Challenges

-   **Missing Tags**: The dataset lacks quotes tagged 'accomplishment', leading to irrelevant retrievals.
-   **LLM Access**: Llama-3 was unavailable, so Google Gemini was used, requiring a `GOOGLE_API_KEY` in `.env`.
-   **Semantic Relevancy**: Low relevancy scores indicate the need for better ground truth alignment or query matching.
-   **Compute Resources**: Fine-tuning and LLM inference benefit from GPU support, available in Kaggle.

## Demo Video

A video walkthrough will be provided (to be uploaded to Google Drive with "Anyone with the link" access). It will demonstrate:

-   Running `notebook/rag.ipynb` for data preparation, fine-tuning, RAG pipeline, and evaluation.
-   Testing example queries and showing outputs.
-   Running `streamlit_app.py` with sample queries and JSON download.
-   Navigating the folder structure and explaining key files.

## Bonus Features

-   **JSON Download**: Implemented in `streamlit_app.py` for query results.
-   **Diverse Queries**: `utils.py` generates varied query phrasings for robust retrieval.
-   **Potential for Multi-Hop Queries**: Query generation can be extended for multi-tag queries (e.g., "quotes tagged with both 'life' and 'love'").

## Submission

-   **Files**:
    -   `notebook/rag.ipynb`: Main notebook.
    -   `utils.py`: Utility functions.
    -   `streamlit_app.py`: Streamlit app.
    -   `model/preprocessed_quotes.csv`: Preprocessed dataset.
    -   `model/fine_tuned_model/`: Fine-tuned model.
    -   `model/faiss_index/`: FAISS vector store.
    -   `model/rag_evaluation_with_semantic_scores.csv`: Evaluation results.
    -   `model.zip`: Zipped model and index files.
    -   `requirements.txt`: Dependencies.
    -   `.env`: Environment variables (not included in submission).
    -   `README.md`: This file.
-   **Video**: To be uploaded to Google Drive with a shared link.
-   **Instructions**: Upload all files (except `.env`) to a Google Drive folder and share the link with appropriate permissions.

## Notes

-   Ensure `GOOGLE_API_KEY` is set in `.env` for Gemini API access.
-   Use `faiss-gpu` if running on a GPU-enabled system for faster vector search.
-   The dataset has 2505 quotes after preprocessing, ensuring robust training and retrieval.

## 👨‍💻 Author

Akash Koley  
📧 [akoley012@gmail.com](mailto:akoley012@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/akashkoley) • [GitHub](https://github.com/AkashKoley012)
