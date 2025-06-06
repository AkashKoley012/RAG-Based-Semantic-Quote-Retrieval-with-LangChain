from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from sentence_transformers import SentenceTransformer, util
from langchain_huggingface import HuggingFaceEmbeddings


def setup_rag_pipeline():
    """Set up the RAG pipeline with LangChain."""
    # Load the same embedding model used during saving
    embedding_model = HuggingFaceEmbeddings(model_name="model/fine_tuned_model")  # e.g., "all-MiniLM-L6-v2"

    # Load the saved FAISS index
    vector_store = FAISS.load_local("model/faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template(
        """Answer the user's query based on the context below.
        If possible, return a **direct quote** from the context.
        
        <context>
        {context}
        </context>
        
        Question: {input}
        Answer:"""
    )
    
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)
    return rag_chain

def query_rag(rag_chain, query):
    """Query the RAG pipeline and return structured output."""
    result = rag_chain.invoke({"input": query})

    # Load the same embedding model used during saving
    embedding_model = SentenceTransformer("model/fine_tuned_model")  # e.g., "all-MiniLM-L6-v2"
    
    context_docs = result.get("context", []) if isinstance(result, dict) else []
    answer = result['answer'] if isinstance(result, dict) else result

    emb_a = embedding_model.encode(answer, convert_to_tensor=True)
    emb_q = embedding_model.encode(query, convert_to_tensor=True)
    sim_score = util.cos_sim(emb_a, emb_q).item()

    response = {
        "answer": answer,
        "source_quotes": [
            {
                "quote": doc.metadata.get('quote', ''),
                "author": doc.metadata.get('author', ''),
                "tags": doc.metadata.get('tags', ''),
                "similarity_score": sim_score
            }
            for doc in context_docs
        ]
    }
    return response