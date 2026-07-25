# CSV Agent – Titanic Dataset
This repository contains a lightweight CSV Agent example based on the **Titanic dataset**.  
It demonstrates how to prepare data, run simple evaluations and interact with the dataset using minimal Python scripts.

## Project Structure
- `main`  
  Entry point for running the agent.
- `mini_eval`  
  Minimal evaluation script to test basic functionality.
- `prepare_data`  
  Data preprocessing and preparation utilities.
- `titanic.xlsx`  
  Example dataset (Titanic passenger data).

## Getting Started

### 1. Clone the repository
```bash
git clone [https://github.com/USERNAME/csv-agent.git](https://github.com/USERNAME/csv-agent.git)
cd csv-agent
```

### 2. Create and activate a virtual environment

**On Windows (cmd):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up Ollama (for local LLM)
Ensure [Ollama](https://ollama.com/) is installed and running, then pull the required model:
```bash
ollama pull llama3.1:8b
```

*(Optional: If you prefer OpenAI instead of Ollama, set your API key: `set OPENAI_API_KEY=your_key` on Windows or `export OPENAI_API_KEY=your_key` on macOS/Linux).*

### 5. Run the project
Generate the data first, then run the agent or evaluation:

```bash
python prepare_data.py
python main.py
python mini_eval.py
```
