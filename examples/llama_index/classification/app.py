from pathlib import Path
from typing import Literal

from continuous_eval.eval.logger import PipelineLogger
from llama_index.core.program import LLMTextCompletionProgram
from pydantic import BaseModel

from examples.llama_index.classification.pipeline import pipeline


class SentimentAnalysis(BaseModel):
    """Sentiment of a news article title."""

    sentiment: Literal["positive", "negative", "neutral"]


prompt_template_str = """Given the title of a news article, say if the the article express a "positive", "negative" or "neutral" sentiment. \
Title: {title}.\nSentiment:"""
program = LLMTextCompletionProgram.from_defaults(
    output_cls=SentimentAnalysis,
    prompt_template_str=prompt_template_str,
    verbose=False,
)


def sentiment_analysis(title: str) -> SentimentAnalysis:
    return program(title=title).sentiment


if __name__ == "__main__":
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    pipelog = PipelineLogger(pipeline=pipeline)
    for datum in pipelog.pipeline.dataset.data:
        title = datum["title"]
        # Retriever
        sentiment = sentiment_analysis(title)
        pipelog.log(uid=datum["uid"], module="sentiment_analysis", value=sentiment)
        print(f"Title: {title}\nA: {sentiment}\n")

    pipelog.save(output_dir/"llamaindex_classification.jsonl")
