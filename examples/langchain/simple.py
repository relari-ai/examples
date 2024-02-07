from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from relari import RelariClient, Tag

client = RelariClient()

chroma_db_file = (
    "/Users/antonap/relari/continuous-eval/data/graham_essays/small/chromadb"
)
db = Chroma(
    persist_directory=chroma_db_file,
    embedding_function=OpenAIEmbeddings(),
)

# Setup RAG pipeline
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
model = ChatGoogleGenerativeAI(model="gemini-pro")

# session = Session()


def rag_pipeline(q):
    # Retrieval
    retrieved_docs = [doc.page_content for doc in retriever.invoke(q)]
    client.log(Tag.RETRIEVED_CONTEXTS, retrieved_docs)
    # Generation
    system_prompt = (
        "You are and expert of the life of Paul Graham.\n"
        "Answer the question below based on the context provided."
    )
    user_prompt = f"Question: {q}\n\n"
    user_prompt += "Contexts:\n" + "\n".join(retrieved_docs)
    try:
        result = model.invoke(system_prompt + user_prompt).content
    except Exception as e:
        print(e)
        result = "Sorry, I cannot answer this question."
    client.log(Tag.GENERATED_ANSWER, result)
    return result


client.set_experiment("Simple RAG")
client.set_dataset("paul_graham")
client.set_metrics(
    {
        "retrieval": [
            {
                "class": "PrecisionRecallF1",
                "params": {
                    "matching_strategy": {"class": "ExactChunkMatch"},
                },
            },
            {
                "class": "RankedRetrievalMetrics",
                "params": {
                    "matching_strategy": {"class": "ExactChunkMatch"},
                },
            },
        ],
    }
)
client.start(run_name=None, params={"k": 3, "dataset": "paul_graham"})

for item in client.iterate_dataset():
    q = item["question"]
    print(f"Q: {q}")
    print(f"A: {rag_pipeline(q)}")
    print("--------------------------------------------------")

client.stop()
