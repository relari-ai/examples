# continuous-eval examples

This repo contains end-to-end examples of GenAI/LLM applications and evaluation pipelines set up using continuous-eval.

Checkout [continuous-eval repo](https://github.com/relari-ai/continuous-eval) and [documentation](https://docs.relari.ai/v0.3/) for more information.

## Examples

| Example Name               | App Framework | Eval Framework  | Description                                       |
| -------------------------- | ------------- | --------------- | ------------------------------------------------- |
| Simple RAG                 | Langchain     | continuous-eval | Simple QA chatbot over select Paul Graham essays  |
| Complex RAG                | Langchain     | continuous-eval | Complex QA chatbot over select Paul Graham essays |
| ReAct Agent                | LlamaIndex    | continuous-eval | QA over Uber financial dataset using agents       |
| Sentiment Classification   | LlamaIndex    | continuous-eval | Single label classification of sentence sentiment |
| Simple RAG                 | Haystack      | continuous-eval | Simple QA chatbot over select Paul Graham essays  |
| Customer Support           | OpenAI Swarm  | continuous-eval | Customer support agent using tools                |

## Installation

In order to run the examples, you need to have Python 3.11 (suggested) and Poetry installed. 
Then, clone this repo and install the dependencies:

```bash
git clone https://github.com/relari-ai/examples.git && cd examples
poetry use 3.11
poetry install --with haystack --with langchain --with llama-index
```

Note that the `--with` flags are optional and only needed if you want to run the examples for the respective frameworks.

## Get started

Each example is in a subfolder: `examples/<FRAMEWORK>/<APP_NAME>/`.

Some examples have just one script to execute (e.g. Haystack's Simple RAG), some have multiple:

- `pipeline.py` defines the application pipeline and the evaluation metrics / tests.
- `app.py` contains the LLM application. Run this script to get the outputs (saved as `results.jsonl`)
- `eval.py` runs the metrics / tests defined by `pipeline.py` (saved as `metrics_results.json` and `test_results.json`)

### Running the examples

To run the examples, you can use the following command:

```bash
poetry run python -m examples.<FRAMEWORK>.<APP_NAME>.app
```

for example `poetry run python3 -m examples.swarm.customer_support.eval`.

To run the evaluation metrics and tests, use:

```bash
poetry run python3 -m examples.<FRAMEWORK>.<APP_NAME>.eval
```

### Additional notes

Depending on the application, the source data for the application (documents and embeddings in Chroma vectorstore) and evaluation (golden dataset) is also provided. Note that for the evaluation golden dataset, there are always two files:

- `dataset.jsonl` contains the inputs (questions) and reference module outputs (ground truths)
- `manifest.yaml` defines the structure of the dataset for the evaluators.

Tweak metrics and tests in `pipeline.py` to try out different metrics.
