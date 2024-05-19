from continuous_eval.eval import Module, Pipeline, Dataset, ModuleOutput
from continuous_eval.metrics.retrieval import (
    # === Deterministic
    PrecisionRecallF1,
    RankedRetrievalMetrics,
)
from continuous_eval.metrics.generation.text import (
    # === Deterministic
    DeterministicFaithfulness,
    DeterministicAnswerCorrectness,
    FleschKincaidReadability,
    # === LLM-based
    LLMBasedAnswerCorrectness,
    LLMBasedAnswerRelevance,
    LLMBasedStyleConsistency,
)
from typing import List, Dict
from continuous_eval.eval.tests import GreaterOrEqualThan, MeanGreaterOrEqualThan

dataset = Dataset("data/paul_graham/dataset")

Documents = List[Dict[str, str]]
DocumentsContent = ModuleOutput(lambda x: [z["page_content"] for z in x])

base_retriever = Module(
    name="base_retriever",
    input=dataset.question,
    output=Documents,
    eval=[
        PrecisionRecallF1().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_context,
        ),
        RankedRetrievalMetrics().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_context,
        ),
    ],
    tests=[
        GreaterOrEqualThan(
            test_name="Context Recall", metric_name="context_recall", min_value=0.9
        ),
    ],
)

bm25_retriever = Module(
    name="bm25_retriever",
    input=dataset.question,
    output=Documents,
    eval=[
        PrecisionRecallF1().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_context,
        ),
        RankedRetrievalMetrics().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_context,
        ),
    ],
    tests=[
        GreaterOrEqualThan(
            test_name="Context Recall", metric_name="context_recall", min_value=0.9
        ),
    ],
)

hyde_generator = Module(
    name="HyDE_generator",
    input=dataset.question,
    output=str,
)

hyde_retriever = Module(
    name="HyDE_retriever",
    input=hyde_generator,
    output=Documents,
    eval=[
        PrecisionRecallF1().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_context,
        ),
        RankedRetrievalMetrics().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_context,
        ),
    ],
    tests=[
        MeanGreaterOrEqualThan(
            test_name="Context Recall", metric_name="context_recall", min_value=0.9
        ),
    ],
)


reranker = Module(
    name="cohere_reranker",
    input=(base_retriever, hyde_retriever, bm25_retriever),
    output=Documents,
    eval=[
        PrecisionRecallF1().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_context,
        ),
        RankedRetrievalMetrics().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_context,
        ),
    ],
    tests=[
        MeanGreaterOrEqualThan(
            test_name="Context Recall", metric_name="context_recall", min_value=0.9
        ),
    ],
)

llm = Module(
    name="answer_generator",
    input=reranker,
    output=str,
    eval=[
        FleschKincaidReadability().use(answer=ModuleOutput()),
        DeterministicAnswerCorrectness().use(
            answer=ModuleOutput(), ground_truth_answers=dataset.ground_truth
        ),
        DeterministicFaithfulness().use(
            answer=ModuleOutput(),
            retrieved_context=ModuleOutput(DocumentsContent, module=reranker),
        ),
        LLMBasedAnswerCorrectness().use(
            question=dataset.question,
            answer=ModuleOutput(),
            ground_truth_answers=dataset.ground_truth,
        ),
        LLMBasedAnswerRelevance().use(
            question=dataset.question,
            answer=ModuleOutput(),
        ),
        LLMBasedStyleConsistency().use(
            answer=ModuleOutput(), ground_truth_answers=dataset.ground_truth
        ),
    ],
    tests=[
        GreaterOrEqualThan(
            test_name="Readability", metric_name="flesch_reading_ease", min_value=20.0
        ),
        GreaterOrEqualThan(
            test_name="Answer Correctness", metric_name="rouge_l_recall", min_value=0.8
        ),
    ],
)

pipeline = Pipeline([base_retriever, hyde_generator, hyde_retriever, bm25_retriever, reranker, llm], dataset=dataset)

print(pipeline.graph_repr())