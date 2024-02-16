from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from relari import RelariClient, Tag
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm

client = RelariClient()

source_files = Path(
    # "../dataset-curator/hosted_data/mini-squad-qa/source_files/"
    "../mlflow-integration/core/data/finance_bench/source_files"
)
chroma_db_folder = Path("data/fin_bench")

search_params = {"k": 5, "chunk_size": 500}

# Initialize Chroma DB
# print("Loading documents")
# loader = DirectoryLoader(source_files, glob="**/*.pdf")
# documents = loader.load()
# print(f"Loaded {len(documents)} documents, Splitting documents")
# text_splitter = CharacterTextSplitter(chunk_size=search_params["chunk_size"], chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# print(f"Created {len(docs)} chunks, Embedding documents")
# db = Chroma.from_documents(
#     docs, OpenAIEmbeddings(), persist_directory=str(chroma_db_folder)
# )
# print("Done")

# Load db from file
db = Chroma(
    persist_directory=str(chroma_db_folder),
    embedding_function=OpenAIEmbeddings(),
)


# Setup RAG pipeline
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": search_params["k"]})

client.set_experiment("Retrieval Experiment")
client.set_dataset("finance_bench")
client.set_metrics(
    {
        "retrieval": [
            {
                "class": "PrecisionRecallF1",
                "params": {
                    "matching_strategy": {"class": "RougeChunkMatch"},
                },
            },
            {
                "class": "RankedRetrievalMetrics",
                "params": {
                    "matching_strategy": {"class": "RougeChunkMatch"},
                },
            },
        ],
    }
)
client.start(run_name=None, params=search_params)

for item in tqdm(client.iterate_dataset(), total=client.dataset_size):
    q = item["question"]
    retrieved_docs = [doc.page_content for doc in retriever.invoke(q)]
    client.log(Tag.RETRIEVED_CONTEXTS, retrieved_docs) 

client.stop()
