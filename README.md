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

1. Clone the repository
2. Create a virtual environment & install dependencies
3. Run the agent

```bash git clone https://github.com/USERNAME/csv-agent.git
cd csv-agent

python -m venv venv source venv/bin/activate
# On Linux/Mac venv\Scripts\activate
# On Windows pip install -r requirements.txt

python main.py
