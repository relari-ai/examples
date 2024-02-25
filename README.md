# continuous-eval examples

This repo contains end-to-end examples of GenAI/LLM applications and evaluation pipelines set up using continuous-eval.

Checkout [continuous-eval repo](https://github.com/relari-ai/continuous-eval) and [documentation](https://docs.relari.ai/v0.3/) for more information.


| Example Name                | App Framework   | Eval Framework | Description                                        |
|-----------------------------|-------------|-|----------------------------------------------------|
| Simple RAG                  | Langchain   | continuous-eval | Simple QA chatbot over select Paul Graham essays   |
| Complex RAG                 | Langchain   | continuous-eval | Complex QA chatbot over select Paul Graham essays  |
| Simple Tools                | LlamaIndex  | continuous-eval | Math question solver using simple tools            |
| Context Augmentation Agent  | LlamaIndex  | continuous-eval | QA over Uber financial dataset using agents          |



## Installation
```bash
git clone https://github.com/relari-ai/examples.git && cd examples
poetry install
```

Add LLM API keys in .env (reference .env.example) for select applications.
-   `COHERE API_KEY` for Cohere Rerankers in RAG examples
-   `GOOGLE_API_KEY` for all LLM calls

## Get started

In each application folder (`examples/[langchain|llamaindex]/APP_NAME/`), here are the files:

-   `pipeline.py` defines the application pipeline and the evaluation metrics / tests.
-   `app.py` contains the LLM application. Run the application to get the outputs (saved as `results.jsonl`)
-   `eval.py` runs the metrics / tests defined by `pipeline.py` (saved as `metrics_results.json` and `test_results.json`)

Depending on the application, the source data for the application (documents and embeddings in Chroma vectorstore) and evaluation (golden dataset) is also provided. Note that for the evaluation golden dataset, there are always two files:

-   `dataset.jsonl` contains the inputs (questions) and reference module outputs (ground truths)
-   `manifest.yaml` defines the structure of the dataset for the evaluators.

Tweak metrics and tests in `pipeline.py` to try out different metrics.