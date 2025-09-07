# main.py
import os, json
import pandas as pd

# --- 0) CSV laden ---
DF_PATH = "titanic.csv"
df = pd.read_csv(DF_PATH)

# --- 1) Tools definieren ---
# WICHTIG: Tools geben Strings zurück (hier JSON-Strings), damit das LLM klar strukturierte Antworten sieht.
from langchain_core.tools import tool

@tool
def tool_schema(dummy: str) -> str:
    """Gibt Spaltennamen und Datentypen als JSON zurück."""
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    return json.dumps(schema)

@tool
def tool_nulls(dummy: str) -> str:
    """Gibt Spalten mit Anzahl fehlender Werte als JSON zurück (nur Spalten mit >0 Missing Values)."""
    nulls = df.isna().sum()
    result = {col: int(n) for col, n in nulls.items() if n > 0}
    return json.dumps(result)

@tool
def tool_describe(input_str: str) -> str:
    """
    Gibt describe()-Statistiken zurück.
    Optional: input_str kann eine komma-separierte Spaltenliste enthalten, z.B. "age, fare".
    """
    cols = None
    if input_str and input_str.strip():
        cols = [c.strip() for c in input_str.split(",") if c.strip() in df.columns]
    stats = df[cols].describe() if cols else df.describe()
    # describe() hat Multi-Index. Fürs LLM flach & lesbar machen:
    return stats.to_csv(index=True)

# --- 2) Tools für LangChain verdrahten ---
tools = [tool_schema, tool_nulls, tool_describe]

# --- 3) LLM konfigurieren ---
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

if USE_OPENAI:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
else:
    from langchain_community.chat_models import ChatOllama
    llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

# --- 4) Schmale Policy/Prompt (Agent-Verhalten) ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = (
    "Du bist ein datenfokussierter Assistent. "
    "Wenn eine Frage Informationen aus der CSV erfordert, nutze zuerst ein passendes Tool. "
    "Nutze pro Schritt nur einen Tool-Aufruf, wenn möglich. "
    "Antworte kompakt und strukturiert. "
    "Wenn kein Tool passt, erkläre kurz warum.\n\n"
    "Verfügbare Tools:\n{tools}\n"
    "Nutze ausschließlich diese Tools: {tool_names}."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

_tool_desc = "\n".join(f"- {t.name}: {t.description}" for t in tools)
_tool_names = ", ".join(t.name for t in tools)
prompt = prompt.partial(tools=_tool_desc, tool_names=_tool_names)

# --- 5) Tool-Calling-Agent erstellen & ausführen ---
from langchain.agents import create_tool_calling_agent, AgentExecutor

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,   # optional: True für Debug-Logs
    max_iterations=3,
)

# --- Um die Datei als Modul nutzen zu können ---
def ask_agent(query: str) -> str:
    return agent_executor.invoke({"input": query})["output"]

if __name__ == "__main__":
    user_query = "Welche Spalten haben Missing Values? Liste 'Spalte: Anzahl'."
    print("\n=== AGENT ANSWER ===")
    print(ask_agent(user_query))
