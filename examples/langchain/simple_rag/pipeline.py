from relari.eval import Module, Pipeline, Dataset
from typing import List, Dict

dataset = Dataset("examples/langchain/rag_data/eval_golden_dataset")

Documents = List[Dict[str, str]]

retriever = Module(
    name="retriever",
    input=dataset.question,
    output=Documents,
)

reranker = Module(
    name="reranker",
    input=retriever,
    output=Documents,
)

llm = Module(
    name="llm",
    input=reranker,
    output=str,
)

pipeline = Pipeline([retriever, reranker, llm], dataset=dataset)

print(pipeline.graph_repr())