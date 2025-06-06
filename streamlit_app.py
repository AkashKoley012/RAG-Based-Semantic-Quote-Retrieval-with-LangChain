import streamlit as st
from utils import setup_rag_pipeline, query_rag
from dotenv import load_dotenv
import json
import os

# Load environment variables from .env file
load_dotenv()

def main():
    st.title("Semantic Quote Retrieval System")
    st.write("Enter a query to retrieve relevant quotes, authors, and tags.")
    
    # Set up RAG pipeline
    qa_chain = setup_rag_pipeline()
    
    # User input
    query = st.text_input("Enter your query (e.g., 'Quotes about courage by women authors')")
    
    if st.button("Search"):
        if query:
            # Get RAG response
            result = query_rag(qa_chain, query)
            
            # Display results
            st.subheader("Answer")
            st.write(result['answer'])
            
            st.subheader("Source Quotes")
            for source in result['source_quotes']:
                st.write(f"**Quote**: {source['quote']}")
                st.write(f"**Author**: {source['author']}")
                st.write(f"**Tags**: {source['tags']}")
                st.write(f"**Similarity Score**: {source['similarity_score']:.2f}")
                st.write("---")
            
            # Provide JSON download
            json_result = json.dumps(result, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_result,
                file_name="query_result.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()