from typing import Dict, List

from continuous_eval.eval import Dataset, Module, ModuleOutput, Pipeline
from continuous_eval.eval.tests import GreaterOrEqualThan, MeanGreaterOrEqualThan
from continuous_eval.metrics.generation.text import (
    DeterministicAnswerCorrectness,
    # === Deterministic
    DeterministicFaithfulness,
    # === LLM-based
    AnswerCorrectness,
    AnswerRelevance,
    StyleConsistency,
)
from continuous_eval.metrics.retrieval import (
    # === Deterministic
    PrecisionRecallF1,
    RankedRetrievalMetrics,
)


Documents = List[Dict[str, str]]


def get_documents_content(x: Documents) -> List[str]:
    return [z["page_content"] for z in x]


DocumentsContent = ModuleOutput(get_documents_content)

dataset = Dataset("data/paul_graham/dataset")

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
        DeterministicFaithfulness().use(
            answer=ModuleOutput(),
            retrieved_context=ModuleOutput(DocumentsContent, module=reranker),
        ),
        AnswerCorrectness().use(
            question=dataset.question,
            answer=ModuleOutput(),
            ground_truth_answers=dataset.ground_truth,
        ),
        StyleConsistency().use(
            answer=ModuleOutput(), ground_truth_answers=dataset.ground_truth
        ),
    ],
    tests=[
        GreaterOrEqualThan(
            test_name="Answer Faithfulness",
            metric_name="token_overlap_faithfulness",
            min_value=0.8,
        ),
        GreaterOrEqualThan(
            test_name="Answer Correctness", metric_name="correctness", min_value=0.8
        ),
        GreaterOrEqualThan(
            test_name="Style Consistency",
            metric_name="consistency",
            min_value=0.8,
        ),
    ],
)

pipeline = Pipeline(
    [base_retriever, hyde_generator, hyde_retriever, bm25_retriever, reranker, llm],
    dataset=dataset,
)

if __name__ == "__main__":
    print(pipeline.graph_repr())
