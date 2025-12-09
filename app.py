from agno.agent import Agent
from agno.models.groq import Groq
from agno.db.sqlite import SqliteDb
from agno.tools.csv_toolkit import CsvTools
from agno.tools.file import FileTools
from agno.tools.pandas import PandasTools
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

data_path = Path(__file__).parent / "data" / "car_details.csv"

base_dir = Path(__file__).parent

db = SqliteDb(db_file="memory.db",
              session_table="session_table")

model = Groq(id="qwen/qwen3-32b")

data_loader_agent = Agent(
    id="data_loader_agent",
    name="Data Loader Agent",
    description="An agent that loads csv files",
    model=model,
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    instructions=["You are expert in loading the CSV files from the project data folder.",
                  "You have access to tools that can help you load and list all the CSV files.",
                  "Make sure to not read more than 20 to 30 rows inside the csv files to avoid memory issues.",
                  "You can also search for the files inside the project folder.",
                  "You have capability to list down files inside the project folder.",
                  "When needed you can read and write files inside the project folder.",
                  "Make sure to read the csv files using only the csv tool provided to you.",
                  "You must always use the exact CSV filename including extension like car_details.csv."],
    tools=[CsvTools(enable_query_csv_file=False,csvs=[data_path]),
           FileTools(base_dir=base_dir)]
)

file_manager_agent = Agent(
    id="file_manager_agent",
    name="File Manager Agent",
    description="An agent that manages files in the project directory",
    model=model,
    instructions=["You are expert File Management Agent",
                  "You're task is to list down files when asked to do so.",
                  "You can also read and write files inside the project folder when required.",
                  "Make sure to never read the csv files, you can only list them."],
    tools=[FileTools(base_dir=base_dir)]
)

data_understanding_agent = Agent(
    id="data-understanding-agent",
    name="Data Understanding Agent",
    model=model,
    role="Data understanding and Exploration assistant",
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    read_chat_history=True,
    instructions=["You are an expert in handling pandas operations on a df",
                  "you can create df and also perform operations on it",
                  "the operations you perform on a df are head(), tail(), info(), describe() for numerical columns and you can also calculate the value_counts() for categorical columns",
                  "make sure to list down the numerical and categorical columns in the df",
                  "you can also check the shape of the df using the .shape attribute",
                  "you also have access to tools which can search for data files"
                  ],
    tools=[PandasTools(), FileTools(base_dir=base_dir)],
)

if __name__ == "__main__":
    data_understanding_agent.cli_app()
