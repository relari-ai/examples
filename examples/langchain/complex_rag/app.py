import os
from pathlib import Path
from continuous_eval.eval.logger import PipelineLogger
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
from examples.langchain.complex_rag.pipeline import pipeline
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()


# Load documents and split
loader = DirectoryLoader("data/paul_graham/documents/208_219_graham_essays")
docs = loader.load()
TextSplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
split_docs = TextSplitter.split_documents(docs)

# Set up Vectorstore
db = Chroma(
    persist_directory=str("examples/langchain/rag_data/vectorstore/208_219_chroma_db"),
    embedding_function=OpenAIEmbeddings(),
)

# Set up LLM
model = ChatGoogleGenerativeAI(model="gemini-pro")


def base_retrieve(q):
    # Basic retriever
    basic_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return basic_retriever.invoke(q)


def bm25_retrieve(q):
    # bm25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents=docs)
    bm25_retriever.k = 2
    return bm25_retriever.invoke(q)


def hyde_generator(q):
    # HyDE generator
    system_prompt = "Generate a hypothetical document paragraph that contains an answer to the question below."
    user_prompt = f"Question: {q}\n\n"
    try:
        result = model.invoke(system_prompt + user_prompt).content
    except Exception as e:
        print(f"{e} unable to generate hypothetical document, using question as HyDE")
        result = q
    return result


def hyde_retrieve(hypothetical_doc):
    # HyDE retriever
    hyde_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return hyde_retriever.invoke(hypothetical_doc)


@retry(stop=stop_after_attempt(10), wait=wait_fixed(61))  # Free tier rate limit
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
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    pipelog = PipelineLogger(pipeline=pipeline)
    for datum in pipeline.dataset.data:
        q = datum["question"]
        # Base Retriever
        base_retrieved_docs = base_retrieve(q)
        pipelog.log(
            uid=datum["uid"],
            module="base_retriever",
            value=[doc.__dict__ for doc in base_retrieved_docs],
        )
        # BM25 Retriever
        bm25_retrieved_docs = bm25_retrieve(q)
        pipelog.log(
            uid=datum["uid"],
            module="bm25_retriever",
            value=[doc.__dict__ for doc in bm25_retrieved_docs],
        )
        # HyDE Generator
        hypothetical_doc = hyde_generator(q)
        pipelog.log(uid=datum["uid"], module="HyDE_generator", value=hypothetical_doc)
        # HyDE Retriever
        hyde_retrieved_docs = hyde_retrieve(hypothetical_doc)
        pipelog.log(
            uid=datum["uid"],
            module="HyDE_retriever",
            value=[doc.__dict__ for doc in hyde_retrieved_docs],
        )

        # Reranker
        fused_docs = base_retrieved_docs + bm25_retrieved_docs + hyde_retrieved_docs
        reranked_docs = rerank(q, fused_docs)
        pipelog.log(
            uid=datum["uid"],
            module="cohere_reranker",
            value=[doc.__dict__ for doc in reranked_docs],
        )
        
        # Generator
        response = ask(q, reranked_docs)
        pipelog.log(uid=datum["uid"], module="answer_generator", value=response)
        print(f"Q: {q}\nA: {response}\n")

    pipelog.save(output_dir / "langchain_complex_rag.jsonl")