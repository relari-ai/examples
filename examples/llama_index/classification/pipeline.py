from continuous_eval.eval import Dataset, ModuleOutput, SingleModulePipeline
from continuous_eval.metrics.classification import SingleLabelClassification

dataset = Dataset("examples/llama_index/classification/data/dataset.jsonl")

pipeline = SingleModulePipeline(
    name="sentiment_analysis",
    dataset=dataset,
    eval=[
        SingleLabelClassification(classes={"positive", "negative", "neutral"}).use(
            predicted_class=ModuleOutput(),
            ground_truth_class=dataset.sentiment,
        ),
    ],
)
