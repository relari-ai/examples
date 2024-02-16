from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from relari import RelariClient, Tag
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm
import json

from continuous_eval.metrics import RougeChunkMatch
from continuous_eval.evaluators import RetrievalEvaluator
from continuous_eval.metrics import PrecisionRecallF1, RankedRetrievalMetrics
from continuous_eval import Dataset



source_files = Path(
    # "../dataset-curator/hosted_data/mini-squad-qa/source_files/"
    "../mlflow-integration/core/data/finance_bench/source_files"
)
chroma_db_folder = Path("data/fin_bench")

search_params = {"k": 5, "chunk_size": 100}

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

dataset_file = Path("/Users/antonap/relari/mlflow-integration/core/data/finance_bench/dataset.jsonl")
dataset = []
with open(dataset_file, "r") as f:
    for line in f:
        dataset.append(json.loads(line))

for i in tqdm(range(len(dataset))):
    q = dataset[i]["question"]
    retrieved_docs = [doc.page_content for doc in retriever.invoke(q)]
    dataset[i]["retrieved_contexts"] = retrieved_docs

# Setup the evaluator
evaluator = RetrievalEvaluator(
    dataset=Dataset(dataset),
    metrics=[
        PrecisionRecallF1(RougeChunkMatch()),
        # RankedRetrievalMetrics(RougeChunkMatch()),
    ],
)
evaluator.run(batch_size=1)
print(evaluator.aggregated_results)
