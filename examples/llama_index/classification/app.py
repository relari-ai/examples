from pathlib import Path
from typing import Literal

from continuous_eval.eval.manager import eval_manager
from llama_index.core.program import LLMTextCompletionProgram
from pydantic import BaseModel

from examples.llama_index.classification.pipeline import pipeline

eval_manager.set_pipeline(pipeline)


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
    eval_manager.start_run()
    while eval_manager.is_running():
        if eval_manager.curr_sample is None:
            break
        title = eval_manager.curr_sample["title"]
        # Retriever
        sentiment = sentiment_analysis(title)
        eval_manager.log("sentiment_analysis", sentiment)
        print(f"Title: {title}\nA: {sentiment}\n")
        eval_manager.next_sample()

    eval_manager.evaluation.save(Path("results.jsonl"))
