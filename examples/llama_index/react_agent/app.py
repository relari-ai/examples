from pathlib import Path
from time import perf_counter
from typing import Any

from continuous_eval.eval.logger import PipelineLogger
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool as _QueryEngineTool
from llama_index.core.tools import ToolMetadata, ToolOutput
from llama_index.llms.openai import OpenAI
from loguru import logger
from tqdm import tqdm

from examples.llama_index.react_agent.pipeline import pipeline

pipelog = PipelineLogger(pipeline=pipeline)
curr_uid = None

llm = OpenAI(model="gpt-4o-mini")
agent_llm = OpenAI(model="gpt-4o-mini")

# We extend Llama-index logger to allow logging
class QueryEngineTool(_QueryEngineTool):
    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        logger.info(
            f"Calling {self.metadata.name} with args: {args} and kwargs: {kwargs}"
        )
        # We log the agent use first...
        pipelog.log(
            uid=curr_uid,
            module="retriever_agent",
            value=self.metadata.name,
            tool_args=kwargs,
        )
        # ...then call the tool...
        ret = super().call(*args, **kwargs)
        # ...and finally log its response
        pipelog.log(uid=curr_uid, module="retriever_agent", value=ret.content)
        # retr_docs = [doc.node.text for doc in ret.raw_output.source_nodes]
        return ret



try:
    # load indexes
    march_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="data/uber/vectorstore/march")
    )
    june_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="data/uber/vectorstore/june")
    )
    sept_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="data/uber/vectorstore/sept")
    )
    print("Indexes loaded...")
except Exception:
    # build indexes across the three data sources
    print("Building indexes...")
    tic = perf_counter()
    # build index
    march_index = VectorStoreIndex.from_documents(
        SimpleDirectoryReader(
            input_files=["data/uber/uber_10q_march_2022.pdf"]
        ).load_data()
    )
    june_index = VectorStoreIndex.from_documents(
        SimpleDirectoryReader(
            input_files=["data/uber/uber_10q_june_2022.pdf"]
        ).load_data()
    )
    sept_index = VectorStoreIndex.from_documents(
        SimpleDirectoryReader(
            input_files=["data/uber/uber_10q_sept_2022.pdf"]
        ).load_data()
    )
    # save index for later use
    march_index.storage_context.persist(persist_dir="data/uber/vectorstore/march")
    june_index.storage_context.persist(persist_dir="data/uber/vectorstore/june")
    sept_index.storage_context.persist(persist_dir="data/uber/vectorstore/sept")
    toc = perf_counter()
    print(f"Indexes built, took {toc - tic:0.4f} seconds.")

# Define tools
query_engine_tools = [
    QueryEngineTool(
        query_engine=march_index.as_query_engine(similarity_top_k=3),
        metadata=ToolMetadata(
            name="uber_march_2022",
            description=(
                "Provides information about Uber quarterly financials ending March 2022"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=june_index.as_query_engine(similarity_top_k=3),
        metadata=ToolMetadata(
            name="uber_june_2022",
            description=(
                "Provides information about Uber quarterly financials ending June 2021"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=sept_index.as_query_engine(similarity_top_k=3),
        metadata=ToolMetadata(
            name="uber_sept_2022",
            description=(
                "Provides information about Uber quarterly financials ending Sept 2021"
            ),
        ),
    ),
]

# Define agent
agent = ReActAgent.from_tools(
    query_engine_tools, llm=agent_llm, verbose=True, max_iterations=20
)


if __name__ == "__main__":
    # agent.chat("Analyze the changes in R&D expenditures and revenue")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print("Running pipeline...")
    for datum in tqdm(pipelog.pipeline.dataset.data):
        curr_uid = datum["uid"]  # set the global variable
        response = agent.chat(datum["question"])
        pipelog.log(uid=curr_uid, module="answer", value=response.response)

    out_fname = output_dir / "llamaindex_react_agent.jsonl"
    pipelog.save(out_fname)
    print("Pipeline run completed.")
    print(f"Results saved to {out_fname}")
