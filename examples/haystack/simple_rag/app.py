import os
import json
from pathlib import Path

from continuous_eval.eval.dataset import Dataset
from continuous_eval.eval.pipeline import ModuleOutput
from continuous_eval.metrics.generation.text import (
    AnswerCorrectness,
    Faithfulness,
)
from continuous_eval.metrics.retrieval import PrecisionRecallF1, RankedRetrievalMetrics
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

# from examples.haystack.utils.conciseness import Conciseness
from examples.haystack.utils.p2p import PipelineEvaluator
from examples.haystack.utils.preprocessor import preprocess_documents

# Set environment variable to disable multiprocessing inside continuous-eval
# Note: this helps to avoid issues with haystack
os.environ["CONTINUOUS_EVAL_DISABLE_MULTIPROCESSING"] = "true" 

_PROMPT_TEMPLATE = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

if __name__ == "__main__":

    # Fetch the Data
    print("Preprocessing document store")
    doc_dir = Path("data/paul_graham/documents/208_219_graham_essays")
    document_store = preprocess_documents(doc_dir)
    print("Done")

    # Building a simple RAG Pipeline
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    retriever = InMemoryEmbeddingRetriever(document_store)
    prompt_builder = PromptBuilder(template=_PROMPT_TEMPLATE)
    generator = OpenAIGenerator(model="gpt-3.5-turbo")

    basic_rag_pipeline = Pipeline()
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", generator)
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder", "llm")

    # Setup evaluation
    # 1. Load an evaluation dataset with synthetic data
    # To do a good evaluation we need a dataset with ground truth, here we use a dataset with synthetic data
    dataset = Dataset("data/paul_graham/dataset/")

    # 2. Build an evaluation pipeline out of the Haystack's pipeline
    # Continuous-eval (similarly to haystack) uses the concept of pipeline.
    # We convert Haystack's pipeline to a continuous-eval pipeline
    eval_pipeline = PipelineEvaluator(basic_rag_pipeline)

    # 3. Specify metrics for each module of the pipeline
    # One core strength of continuous-eval is the MODULAR EVALUATION.
    # What we mean is that we can easily specify metrics for each module of the pipeline.
    # 3.0. Each module return arbitrary data structure, here we use a LambdaField to specify how to extract the data
    def extract_llm_answer(x):
        # the llm answer is in `replies`
        return x["replies"][0]


    llm_answer = ModuleOutput(module="llm", selector=extract_llm_answer)


    def extract_retrieved_context(x):
        # the content of the retrieved chunks
        return [z.content for z in x["documents"]]


    retrieved_context = ModuleOutput(module="retriever", selector=extract_retrieved_context)

    # 3.1 For each module of interest we specify the metrics
    eval_pipeline.add_metrics(
        "llm",
        [
            Faithfulness().use(
                answer=llm_answer,
                retrieved_context=retrieved_context,
                question=dataset.question,
            ),
            # Conciseness.use(  # This is a custom metric
            #     question=dataset.question,
            #     answer=llm_answer,
            #     ground_truth_answers=dataset.ground_truth,
            # ),
            AnswerCorrectness().use(  # This is a deterministic metric
                question=dataset.question,
                answer=llm_answer,
                ground_truth_answers=dataset.ground_truth,
            ),
        ],
    )
    eval_pipeline.add_metrics(
        "retriever",
        [
            PrecisionRecallF1().use(  # Deterministic metric
                retrieved_context=retrieved_context,
                ground_truth_context=dataset.ground_truth_context,
            ),
            RankedRetrievalMetrics().use(  # Deterministic metric
                retrieved_context=retrieved_context,
                ground_truth_context=dataset.ground_truth_context,
            ),
        ],
    )

    # 4. Run the evaluation
    # Similar to haystack, we need to specify the input map for the pipeline
    # But here we can use directly dataset fields as input
    input_map = {
        "text_embedder": {"text": dataset.question},
        "prompt_builder": {"question": dataset.question},
    }
    metrics_results = eval_pipeline.run_evaluation(dataset, input_map)

    # 5. Print the results
    # The results are stored in a dictionary-like object
    # aggregated() method will aggregate the results printing the averages
    print(json.dumps(metrics_results.aggregate(), indent=4))
