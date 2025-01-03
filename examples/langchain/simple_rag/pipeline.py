from typing import Dict, List

from continuous_eval.eval import Dataset, Module, ModuleOutput, Pipeline
from continuous_eval.eval.tests import MeanGreaterOrEqualThan
from continuous_eval.metrics.generation.text import (
    DeterministicAnswerCorrectness,
    FleschKincaidReadability,
)
from continuous_eval.metrics.retrieval import PrecisionRecallF1, RankedRetrievalMetrics

Documents = List[Dict[str, str]]


def get_documents_content(x: Documents) -> List[str]:
    return [z["page_content"] for z in x]


DocumentsContent = ModuleOutput(get_documents_content)

dataset = Dataset("data/paul_graham/dataset")

retriever = Module(
    name="retriever",
    input=dataset.question,
    output=Documents,
    eval=[
        PrecisionRecallF1().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_context,
        ),
    ],
    tests=[
        MeanGreaterOrEqualThan(
            test_name="Average Precision", metric_name="context_recall", min_value=0.8
        ),
    ],
)

reranker = Module(
    name="reranker",
    input=retriever,
    output=Documents,
    eval=[
        RankedRetrievalMetrics().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_context,
        ),
    ],
    tests=[
        MeanGreaterOrEqualThan(
            test_name="Context Recall", metric_name="average_precision", min_value=0.7
        ),
    ],
)

llm = Module(
    name="llm",
    input=reranker,
    output=str,
    eval=[
        FleschKincaidReadability().use(answer=ModuleOutput()),
        DeterministicAnswerCorrectness().use(
            answer=ModuleOutput(), ground_truth_answers=dataset.ground_truth
        ),
    ],
)

pipeline = Pipeline([retriever, reranker, llm], dataset=dataset)

if __name__ == "__main__":
    print(pipeline.graph_repr())
