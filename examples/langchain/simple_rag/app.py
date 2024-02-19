import os
from pathlib import Path

from continuous_eval.eval.manager import eval_manager
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from examples.langchain.simple_rag.pipeline import pipeline

load_dotenv()

eval_manager.set_pipeline(pipeline)


# Setup Chroma
db = Chroma(
    persist_directory=str("/Users/yisz/Downloads/208_219_chroma_db"),
    embedding_function=OpenAIEmbeddings(),
)

# Setup RAG pipeline
search_params = {"k": 10, "chunk_size": 500}
retriever = db.as_retriever(
    search_type="similarity", search_kwargs={"k": search_params["k"]}
)
compressor = CohereRerank(cohere_api_key=os.getenv("COHERE_API_KEY"))
model = ChatGoogleGenerativeAI(model="gemini-pro")


def retrieve(q):
    return retriever.invoke(q)


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


def rerank(q, retrieved_docs):
    return compressor.compress_documents(retrieved_docs, q)


if __name__ == "__main__":
    eval_manager.start_run()
    while eval_manager.is_running():
        if eval_manager.curr_sample is None:
            break
        q = eval_manager.curr_sample["question"]
        # Retriever
        retrieved_docs = retrieve(q)
        eval_manager.log("retriever", [doc.__dict__ for doc in retrieved_docs])
        # Reranker
        reranked_docs = rerank(q, retrieved_docs)
        eval_manager.log("reranker", [doc.__dict__ for doc in reranked_docs])
        # Generator
        response = ask(q, retrieved_docs)
        eval_manager.log("llm", response)
        print(f"Q: {q}\nA: {response}\n")
        eval_manager.next_sample()

    eval_manager.save_results(Path("results.jsonl"))
