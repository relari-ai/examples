import os
from pathlib import Path
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from continuous_eval.eval.manager import eval_manager
from examples.langchain.simple_rag.pipeline import pipeline
load_dotenv()

def retrieve(q):
    db = Chroma(
        persist_directory=str("/data/vectorstore/208_219_chroma_db"),
        embedding_function=OpenAIEmbeddings(),
    )
    retriever = db.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    )
    return retriever.invoke(q)

def rerank(q, retrieved_docs):
    compressor = CohereRerank(cohere_api_key=os.getenv("COHERE_API_KEY"))
    return compressor.compress_documents(retrieved_docs, q)

def ask(q, retrieved_docs):
    model = ChatGoogleGenerativeAI(model="gemini-pro")
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
    eval_manager.set_pipeline(pipeline)
    eval_manager.start_run()
    while eval_manager.is_running():
        if eval_manager.curr_sample is None:
            break
        q = eval_manager.curr_sample["question"]
        # Run and log Retriever results
        retrieved_docs = retrieve(q)
        eval_manager.log("retriever", [doc.__dict__ for doc in retrieved_docs])
        # Run and log Reranker results
        reranked_docs = rerank(q, retrieved_docs)
        eval_manager.log("reranker", [doc.__dict__ for doc in reranked_docs])
        # Run and log Generator results
        response = ask(q, reranked_docs)
        eval_manager.log("llm", response)
        print(f"Q: {q}\nA: {response}\n")
        eval_manager.next_sample()

    eval_manager.evaluation.save(Path("results.jsonl"))
