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
    "../dataset-curator/hosted_data/mini-squad-qa/source_files/"
)
chroma_db_folder = Path("data/squad")

# Initialize Chroma DB
# loader = DirectoryLoader(source_files, glob="**/*.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# db = Chroma.from_documents(
#     docs, OpenAIEmbeddings(), persist_directory=str(chroma_db_folder)
# )

# Load db from file
db = Chroma(
    persist_directory=str(chroma_db_folder),
    embedding_function=OpenAIEmbeddings(),
)


search_params = {"k": 5}
# Setup RAG pipeline
retriever = db.as_retriever(search_type="similarity", search_kwargs=search_params)

client.set_experiment("Retrieval Experiment")
client.set_dataset("mini-squad-qa")
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

for item in tqdm(client.iterate_dataset(), total=195):
    q = item["question"]
    retrieved_docs = [doc.page_content for doc in retriever.invoke(q)]
    client.log(Tag.RETRIEVED_CONTEXTS, retrieved_docs) 

client.stop()
