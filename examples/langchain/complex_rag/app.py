import os
from pathlib import Path

from continuous_eval.eval.manager import eval_manager
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
from examples.langchain.complex_rag.pipeline import pipeline

load_dotenv()

eval_manager.set_pipeline(pipeline)

# Load documents and split
loader = DirectoryLoader("data/documents/208_219_graham_essays")
docs = loader.load()
TextSplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
split_docs = TextSplitter.split_documents(docs)

# Set up Ve
db = Chroma(
    persist_directory=str("data/vectorstore/208_219_chroma_db"),
    embedding_function=OpenAIEmbeddings(),
)

# Set up LLM
model = ChatGoogleGenerativeAI(model="gemini-pro")

def base_retrieve(q):
    # Basic retriever
    basic_retriever = db.as_retriever(
        search_type="similarity", search_kwargs={"k": 8}
    )
    return basic_retriever.invoke(q)

def bm25_retrieve(q):
    # bm25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents=docs)
    bm25_retriever.k = 10
    return bm25_retriever.invoke(q)

def hyde_generator(q):
    # HyDE generator
    system_prompt = (
        "Generate a hypothetical document paragraph that contains an answer to the question below."
    )
    user_prompt = f"Question: {q}\n\n"
    try:
        result = model.invoke(system_prompt + user_prompt).content
    except Exception as e:
        print(f"{e} unable to generate hypothetical document, using question as HyDE")
        result = q
    return result

def hyde_retrieve(hypothetical_doc):
    # HyDE retriever
    hyde_retriever = db.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    return hyde_retriever.invoke(hypothetical_doc)

def rerank(q, retrieved_docs):
    # reranker
    compressor = CohereRerank(cohere_api_key=os.getenv("COHERE_API_KEY"))
    compressor.top_n = 5
    return compressor.compress_documents(retrieved_docs, q)


def ask(q, retrieved_docs):
    system_prompt = (
        "You are and expert of the life of Paul Graham.\n"
        "Answer the question below based on the context provided."
    )
    user_prompt = f"Question: {q}\n\n"
    user_prompt += "Contexts:\n" + "\n".join(
        [doc.page_content for doc in retrieved_docs]
    )
    try:
        result = model.invoke(system_prompt + user_prompt).content
    except Exception as e:
        print(e)
        result = "Sorry, I cannot answer this question."
    return result


if __name__ == "__main__":
    eval_manager.start_run()
    while eval_manager.is_running():
        if eval_manager.curr_sample is None:
            break
        q = eval_manager.curr_sample["question"]
        # Base Retriever
        base_retrieved_docs = base_retrieve(q)
        eval_manager.log("base_retriever", [doc.__dict__ for doc in base_retrieved_docs])
        # BM25 Retriever
        bm25_retrieved_docs = bm25_retrieve(q)
        eval_manager.log("bm25_retriever", [doc.__dict__ for doc in bm25_retrieved_docs])
        # HyDE Generator
        hypothetical_doc = hyde_generator(q)
        eval_manager.log("HyDE_generator", hypothetical_doc)
        # HyDE Retriever
        hyde_retrieved_docs = hyde_retrieve(hypothetical_doc)
        eval_manager.log("HyDE_retriever", [doc.__dict__ for doc in hyde_retrieved_docs])

        fused_docs = base_retrieved_docs + bm25_retrieved_docs + hyde_retrieved_docs
        # Reranker
        reranked_docs = rerank(q, fused_docs)
        eval_manager.log("cohere_reranker", [doc.__dict__ for doc in reranked_docs])
        # Generator
        response = ask(q, reranked_docs)
        eval_manager.log("answer_generator", response)
        print(f"Q: {q}\nA: {response}\n")
        eval_manager.next_sample()

    eval_manager.evaluation.save(Path("results.jsonl"))
