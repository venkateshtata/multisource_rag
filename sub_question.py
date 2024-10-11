import nest_asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
import os
from llama_index.core import Settings


nest_asyncio.apply()


Settings.llm = OpenAI(temperature=0.2, model="gpt-3.5-turbo")

lyft_docs = SimpleDirectoryReader(
    input_files=["./data/10k/lyft_2021.pdf"]
).load_data()
uber_docs = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()

lyft_index = VectorStoreIndex.from_documents(lyft_docs)

uber_index = VectorStoreIndex.from_documents(uber_docs)


lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)


query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021"
            ),
        ),
    ),
]

s_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools
)

response = s_engine.query(
    "Compare and contrast the customer segments and geographies that grew the"
    " fastest"
)

print('response: ', response)