import nest_asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
import os
from llama_index.core import Settings
import openai
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
)
from llama_index.core import SQLDatabase
from llama_index.llms.openai import OpenAI
from sqlalchemy import insert
from llama_index.core.query_engine import NLSQLTableQueryEngine



if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running this script.")

openai.api_key = os.environ["OPENAI_API_KEY"]

nest_asyncio.apply()



# Creating SQL Database

engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

# create city SQL table
table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)
metadata_obj.create_all(engine)


llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
sql_database = SQLDatabase(engine, include_tables=["city_stats"])

rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {
        "city_name": "Chicago",
        "population": 2679000,
        "country": "United States",
    },
    {"city_name": "Seoul", "population": 9776000, "country": "South Korea"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

# view current table
stmt = select(
    city_stats_table.c.city_name,
    city_stats_table.c.population,
    city_stats_table.c.country,
).select_from(city_stats_table)

with engine.connect() as connection:
    results = connection.execute(stmt).fetchall()
    print(results)




Settings.llm = OpenAI(temperature=0.2, model="gpt-3.5-turbo")

lyft_docs = SimpleDirectoryReader(
    input_files=["./data/10k/lyft_2021.pdf"]
).load_data()
uber_docs = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()\


lyft_index = VectorStoreIndex.from_documents(lyft_docs)

uber_index = VectorStoreIndex.from_documents(uber_docs)


lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)
sql_engine = NLSQLTableQueryEngine(sql_database=sql_database, tables=["city_stats"], llm=llm)


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
    QueryEngineTool(
        query_engine=sql_engine,
        metadata=ToolMetadata(
            name="sql_table",
            description=(
                "Provides populations informations about various cities across the world"
            ),
        ),
    ),
]



s_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools
)

response = s_engine.query(
    "Compare and contrast the customer segments and geographies that grew the fastest, and also which are the most populated cities among Tokyo and Toronto ?"
)

print('response: ', response)