from continuous_eval.eval import Module, Pipeline, Dataset, ModuleOutput
from continuous_eval.metrics.retrieval import (
    # === Deterministic
    PrecisionRecallF1,
    RankedRetrievalMetrics,
    # === LLM-based
    LLMBasedContextCoverage,
    LLMBasedContextPrecision,
)
from continuous_eval.metrics.generation.text import (
    # === Deterministic
    DeterministicFaithfulness,
    DeterministicAnswerCorrectness,
    FleschKincaidReadability,
    # === Semantic
    BertAnswerRelevance,
    BertAnswerSimilarity,
    DebertaAnswerScores,
    # === LLM-based
    LLMBasedFaithfulness,
    LLMBasedAnswerCorrectness,
    LLMBasedAnswerRelevance,
    LLMBasedStyleConsistency,
)
from typing import List, Dict
from continuous_eval.eval.tests import GreaterOrEqualThan

dataset = Dataset("../mlflow-integration/core/data/paul-graham")

Documents = List[Dict[str, str]]
DocumentsContent = ModuleOutput(lambda x: [z["page_content"] for z in x])

retriever = Module(
    name="retriever",
    input=dataset.question,
    output=Documents,
    eval=[
        PrecisionRecallF1().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_contexts,
        ),
        RankedRetrievalMetrics().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_contexts,
        ),
        LLMBasedContextPrecision().use(
            retrieved_context=DocumentsContent, question=dataset.question, 
        ),
        LLMBasedContextCoverage().use(
            question=dataset.question,
            retrieved_contexts=DocumentsContent,
            ground_truth_answers=dataset.ground_truths,
        ),
    ],
    tests=[
        GreaterOrEqualThan(
            test_name="Context Recall", metric_name="context_recall", min_value=0.9
        ),
    ],
)

reranker = Module(
    name="reranker",
    input=retriever,
    output=Documents,
    eval=[
        PrecisionRecallF1().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_contexts,
        ),
        RankedRetrievalMetrics().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_contexts,
        ),
    ],
    tests=[
        GreaterOrEqualThan(
            test_name="Context Recall", metric_name="context_recall", min_value=0.9
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
            answer=ModuleOutput(), ground_truth_answers=dataset.ground_truths
        ),
        DeterministicFaithfulness().use(
            answer=ModuleOutput(),
            retrieved_contexts=ModuleOutput(DocumentsContent, module=reranker),
        ),
        BertAnswerRelevance().use(answer=ModuleOutput(), question=dataset.question),
        BertAnswerSimilarity().use(
            answer=ModuleOutput(), ground_truth_answers=dataset.ground_truths
        ),
        DebertaAnswerScores().use(
            answer=ModuleOutput(), ground_truth_answers=dataset.ground_truths
        ),
        LLMBasedFaithfulness().use(
            answer=ModuleOutput(),
            retrieved_context=ModuleOutput(DocumentsContent, module=reranker),
            question=dataset.question,
        ),
        LLMBasedAnswerCorrectness().use(
            question=dataset.question,
            answer=ModuleOutput(),
            ground_truth_answers=dataset.ground_truths,
        ),
        LLMBasedAnswerRelevance().use(
            question=dataset.question,
            answer=ModuleOutput(),
        ),
        LLMBasedStyleConsistency().use(
            answer=ModuleOutput(), ground_truth_answers=dataset.ground_truths
        ),
    ],
    tests=[
        GreaterOrEqualThan(
            test_name="Readability", metric_name="flesch_reading_ease", min_value=20.0
        ),
        GreaterOrEqualThan(
            test_name="Answer Correctness", metric_name="rouge_l_f1", min_value=0.8
        ),
    ],
)

pipeline = Pipeline([retriever, reranker, llm], dataset=dataset)
