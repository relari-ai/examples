from continuous_eval.eval import Module, Pipeline, Dataset, ModuleOutput
from continuous_eval.metrics.retrieval import PrecisionRecallF1, RankedRetrievalMetrics # Deterministic metrics
from continuous_eval.metrics.generation.text import (
    FleschKincaidReadability, # Deterministic metric
    DebertaAnswerScores, # Semantic metric
    LLMBasedFaithfulness, # LLM-based metric
)
from typing import List, Dict
from continuous_eval.eval.tests import MeanGreaterOrEqualThan

dataset = Dataset("data/eval_golden_dataset")

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
        DebertaAnswerScores().use(
            answer=ModuleOutput(), ground_truth_answers=dataset.ground_truths
        ),
        LLMBasedFaithfulness().use(
            answer=ModuleOutput(),
            retrieved_context=ModuleOutput(DocumentsContent, module=reranker),
            question=dataset.question,
        ),
    ],
    tests=[
        MeanGreaterOrEqualThan(
            test_name="Deberta Entailment", metric_name="deberta_answer_entailment", min_value=0.5
        ),
    ],
)

pipeline = Pipeline([retriever, reranker, llm], dataset=dataset)

print(pipeline.graph_repr())