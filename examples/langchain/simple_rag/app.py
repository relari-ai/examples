import os
from pathlib import Path

from continuous_eval.eval.logger import PipelineLogger
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm
from examples.langchain.simple_rag.pipeline import pipeline
from langchain_cohere import CohereRerank

load_dotenv()

db = Chroma(
    persist_directory=str("data/paul_graham/vectorstore/208_219_chroma_db"),
    embedding_function=OpenAIEmbeddings(),
)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
compressor = CohereRerank(model="rerank-v3.5", top_n=3)
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=1.0)


def retrieve(q):
    return retriever.invoke(q)


@retry(stop=stop_after_attempt(10), wait=wait_fixed(61))  # Free tier rate limit
def rerank(q, retrieved_docs):
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
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    pipelog = PipelineLogger(pipeline=pipeline)
    for datum in tqdm(pipeline.dataset.data):
        q = datum["question"]
        # Retriever results
        retrieved_docs = retrieve(q)
        pipelog.log(
            uid=datum["uid"],
            module="retriever",
            value=[doc.__dict__ for doc in retrieved_docs],
        )
        # Reranker
        reranked_docs = rerank(q, retrieved_docs)
        pipelog.log(
            uid=datum["uid"],
            module="reranker",
            value=[doc.__dict__ for doc in reranked_docs],
        )
        # Generator
        response = ask(q, reranked_docs)
        pipelog.log(uid=datum["uid"], module="llm", value=response)

    pipelog.save(output_dir / "langchain_simple_rag.jsonl")
