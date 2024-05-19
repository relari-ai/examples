from typing import Dict, List

from continuous_eval.eval import Dataset, Module, ModuleOutput, Pipeline
from continuous_eval.eval.tests import MeanGreaterOrEqualThan
from continuous_eval.metrics.generation.text import (
    DeterministicAnswerCorrectness,
    FleschKincaidReadability,
)
from continuous_eval.metrics.retrieval import PrecisionRecallF1, RankedRetrievalMetrics

dataset = Dataset("data/paul_graham/dataset")

Documents = List[Dict[str, str]]
DocumentsContent = ModuleOutput(lambda x: [z["page_content"] for z in x])

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

print(pipeline.graph_repr())
