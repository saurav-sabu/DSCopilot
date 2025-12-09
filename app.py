from agno.agent import Agent
from agno.models.groq import Groq
from agno.db.sqlite import SqliteDb
from agno.tools.csv_toolkit import CsvTools
from agno.os import AgentOS
from agno.team import Team
from agno.tools.file import FileTools
from agno.tools.pandas import PandasTools
from agno.tools.visualization import VisualizationTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.python import PythonTools
from agno.tools.shell import ShellTools
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
    role = "Data loading and reading the csv files",
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
    role = "Manages the file system",
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
    role = "Data Understanding and exploration assistant",
    model=model,
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    read_chat_history=True,
    search_session_history=True,
    instructions=[
    "You are an expert in handling pandas operations on a df",
    "you can create df and also perform operations on it",
    "the operations you perform on a df are head(), tail(), info(), describe()",
    "make sure to list down the numerical and categorical columns in the df",
    "you can also check the shape of the df using the .shape attribute",
    "you also have access to tools which can search for data files",
    "When using create_pandas_dataframe tool, always pass filepath_or_buffer as an object with key 'path'. Example: {'path':'data/car_details.csv'}"
],
    tools=[PandasTools(), FileTools(base_dir=base_dir)],
)

visualization_agent = Agent(
    id="data-visualization-agent",
    name="Data Visualization Agent",
    role = "Plotting and data visualization assistant",
    model=model,
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    tools=[PandasTools(), FileTools(base_dir=base_dir), VisualizationTools(output_dir="plots")],
    instructions=["You are an expert in creating plots using matplotlib and seaborn libraries.",
                  "You have access to file tools to list down the data files inside the project directory.",
                  "You also have access to pandas tools that can run pandas specific code that will be used as input to create plots.",
                  "You can create different types of plots like bar plots, line plots, scatter plots, histograms, box plots, heatmaps etc.",
                  "You can also customize the plots by adding titles, labels, legends, changing colors and styles.",
                  "Make sure to input the correct data for thre respective plot you want to create."]
)

coding_agent = Agent(
    id="coding_agent",
    name="Coding Agent",
    role = "Python coding assistant for ML and Data Science tasks",
    description="An agent that writes code snippets",
    model=model,
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    read_chat_history=True,
    tools=[PythonTools(base_dir=base_dir),DuckDuckGoTools(), ShellTools(base_dir=base_dir)],
    instructions=["You are an expert coding agent proficient in writing Python code snippets.",
                  "You have access to tools that can help you write, read and execute Python code.",
                  "Your main task is to write code specific to machine learning, data science and data analysis.",
                  "You may be using numpy, pandas, matplotlib, seaborn, scikit-learn and other related libraries.",
                  "You will be used to write python code for ML and Data Science specific tasks such as data cleaning, feature engineering, model training and evaluation.",
                  "You have access to toolthat can list files inside the project directory and read them",
                  "Python Tool also lets you to create python files and write them and also run them to get the desired output.",
                  "You also have access web searc tool to search the latest documentation and code examples online.",
                  "Nake sure to get the code reviewed by the user and only then write it to the file when the user approves it.",
                  "If you want to add packages using shell tool and use the command `uv add <package-name>` to install the package.",
                  "Do not use the shell tool to execute any other command than installing packages."]
)

shell_agent = Agent(
    id="shell_agent",
    name="Shell Agent",
    role = "Shell command execution assistant",
    description="An agent that executes shell commands",
    model=model,
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    read_chat_history=True,
    tools=[ShellTools(base_dir=base_dir)],
    instructions=["You have the capability to execute shell commands inside the project directory.",
                  "Use this capability with extreme caution and only when absolutely necessary.",
                  "Use the shell tool if unable to execute python file.",
                  "Use the command uv run <python-file-name> to execute python files.",
                  "Do not use the shell tool command which can make changes to system or project structure",
                  "Just use it to read the project structure or to run files",
                  "Do not delete or modify any files using shell tool."]
)

data_science_team = Team(
    id="ds_copilot_team",
    name="DS Copilot Team",
    role="Team Leader / Project Manager",
    members=[data_loader_agent,
            file_manager_agent,
            data_understanding_agent,
            visualization_agent,
            coding_agent,
            shell_agent],
    model=model,
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    read_chat_history=True,
    session_state={},
    add_session_state_to_context=True,
    add_member_tools_to_context=True,
    enable_agentic_state=True,
    instructions=["You are an expert data scientist and team leader",
                  "You manage team members who are good at coding using pandas, generating visualization using matplotlib,"
                  "executing shell commands, loading data files and also managing the project file system",
                  "delegate tasks to members according to their expertise and always make sure to review the activity",
                  "Your task is to assist the user in completing their Machine learning pipeline having steps such as data loading, data understanding, data cleaing ,plotting charts, feature engineering, model training and model evaluation",
                  "Whenever you generate code, first get it reviewed by the user before saving and executing it.",
                  "Assist the user at all times and provide steps for each stage so that the user knows what to do",
                  "You have access to the session state where you can add session wise memory.",
                  "Always try to add important stuff such as data path, path to models and path to source files",
                  "whenver the user asks you to remember anything, just add it to session state. so that it can be retrieved later",
                  "Always to assist the user with ideas and always think like a professional data scientist.",
                  "if you get any errors try to have approach to debug and solve issues"]
)

agent_os = AgentOS(
    id="ds_copilot_os",
    name="DS Copilot OS",
    description="This team of agents helps in building a complete data science project",
    teams=[data_science_team]
)

app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(
        app="app:app"
    )