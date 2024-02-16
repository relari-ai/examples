from typing import Any

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


class LoggableQueryEngineTool(QueryEngineTool):
    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        logger.info(
            f"Calling {self.metadata.name} with args: {args} and kwargs: {kwargs}"
        )
        ret = super().call(*args, **kwargs)
        return ret


def main(query, verbose: bool = False):
    try:
        # load indexes
        storage_context = StorageContext.from_defaults(persist_dir="./data/uber/march")
        march_index = load_index_from_storage(storage_context)
        storage_context = StorageContext.from_defaults(persist_dir="./data/uber/june")
        june_index = load_index_from_storage(storage_context)
        storage_context = StorageContext.from_defaults(persist_dir="./data/uber/sept")
        sept_index = load_index_from_storage(storage_context)
    except:
        # build indexes across the three data sources
        march_docs = SimpleDirectoryReader(
            input_files=["./data/uber/uber_10q_march_2022.pdf"]
        ).load_data()
        june_docs = SimpleDirectoryReader(
            input_files=["./data/uber/uber_10q_june_2022.pdf"]
        ).load_data()
        sept_docs = SimpleDirectoryReader(
            input_files=["./data/uber/uber_10q_sept_2022.pdf"]
        ).load_data()
        # build index
        march_index = VectorStoreIndex.from_documents(march_docs)
        june_index = VectorStoreIndex.from_documents(june_docs)
        sept_index = VectorStoreIndex.from_documents(sept_docs)
        # persist index
        march_index.storage_context.persist(persist_dir="./data/uber/march")
        june_index.storage_context.persist(persist_dir="./data/uber/june")
        sept_index.storage_context.persist(persist_dir="./data/uber/sept")

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
        verbose=verbose,
    )
    response = context_agent.chat(query)
    print(response)


if __name__ == "__main__":
    main("What is Uber revenue as of June 2022?")
