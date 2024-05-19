from pathlib import Path
from time import perf_counter
from typing import Any

from continuous_eval.eval.logger import PipelineLogger, LogMode
from llama_index.agent.openai_legacy import ContextRetrieverOpenAIAgent
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.tools.types import ToolMetadata, ToolOutput
from loguru import logger

from examples.llama_index.context_augmentation.pipeline import pipeline

## We setup the pipeline logger
pipelog = PipelineLogger(pipeline=pipeline)
curr_uid = None

VERBOSE = False


## We extend Llama-index logger to allow logging
class LoggableQueryEngineTool(QueryEngineTool):
    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        logger.info(
            f"Calling {self.metadata.name} with args: {args} and kwargs: {kwargs}"
        )
        ret = super().call(*args, **kwargs)
        # we log the agent use first
        pipelog.log(
            uid=curr_uid,
            module="retriever_agent",
            value=self.metadata.name,
            tool_args=kwargs,
        )
        # and then the response
        pipelog.log(uid=curr_uid, module="retriever_agent", value=ret.content)
        # ret.raw_output.source_nodes
        return ret


try:
    # load indexes
    storage_context = StorageContext.from_defaults(
        persist_dir="data/uber/vectorstore/march"
    )
    march_index = load_index_from_storage(storage_context)
    storage_context = StorageContext.from_defaults(
        persist_dir="data/uber/vectorstore/june"
    )
    june_index = load_index_from_storage(storage_context)
    storage_context = StorageContext.from_defaults(
        persist_dir="data/uber/vectorstore/sept"
    )
    sept_index = load_index_from_storage(storage_context)
    print("Indexes loaded...")
except:
    # build indexes across the three data sources
    print("Building indexes...")
    tic = perf_counter()
    march_docs = SimpleDirectoryReader(
        input_files=["data/uber/uber_10q_march_2022.pdf"]
    ).load_data()
    june_docs = SimpleDirectoryReader(
        input_files=["data/uber/uber_10q_june_2022.pdf"]
    ).load_data()
    sept_docs = SimpleDirectoryReader(
        input_files=["data/uber/uber_10q_sept_2022.pdf"]
    ).load_data()
    # build index
    march_index = VectorStoreIndex.from_documents(march_docs)
    june_index = VectorStoreIndex.from_documents(june_docs)
    sept_index = VectorStoreIndex.from_documents(sept_docs)
    # persist index
    march_index.storage_context.persist(persist_dir="data/uber/vectorstore/march")
    june_index.storage_context.persist(persist_dir="data/uber/vectorstore/june")
    sept_index.storage_context.persist(persist_dir="data/uber/vectorstore/sept")
    toc = perf_counter()
    print(f"Indexes built, took {toc - tic:0.4f} seconds.")

march_engine = march_index.as_query_engine(similarity_top_k=3)
june_engine = june_index.as_query_engine(similarity_top_k=3)
sept_engine = sept_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    LoggableQueryEngineTool(
        query_engine=march_engine,
        metadata=ToolMetadata(
            name="uber_march_10q",
            description=(
                "Provides information about Uber 10Q filings for March 2022. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    LoggableQueryEngineTool(
        query_engine=june_engine,
        metadata=ToolMetadata(
            name="uber_june_10q",
            description=(
                "Provides information about Uber financials for June 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    LoggableQueryEngineTool(
        query_engine=sept_engine,
        metadata=ToolMetadata(
            name="uber_sept_10q",
            description=(
                "Provides information about Uber financials for Sept 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

texts = [
    "Abbreviation: FINRA (Financial Industry Regulatory Authority)",
]
docs = [Document(text=t) for t in texts]
context_index = VectorStoreIndex.from_documents(docs)

context_agent = ContextRetrieverOpenAIAgent.from_tools_and_retriever(
    tools=query_engine_tools,
    retriever=context_index.as_retriever(),
    verbose=VERBOSE,
)


def ask(query: str):
    response = context_agent.chat(query)
    return response.response


if __name__ == "__main__":
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print("Running pipeline...")
    for datum in pipelog.pipeline.dataset.data:
        curr_uid = datum["uid"]  # set the global variable
        response = ask(datum["question"])
        pipelog.log(uid=curr_uid, module="answer", value=response)
        print(f"Q: {datum['question']}\nA: {response}\n")

    out_fname = output_dir / "llamaindex_contex_augmentation.jsonl"
    pipelog.save(out_fname)
    print("Pipeline run completed.")
    print(f"Results saved to {out_fname}")
