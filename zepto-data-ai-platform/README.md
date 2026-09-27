# Zepto Data & AI Platform

A modular data and AI platform project containing:

1. Data scraping and SQLite analytics pipeline
2. Titanic EDA and machine learning analysis
3. Zepto policy support assistant using embeddings, ChromaDB, LangGraph and FastAPI

---

## Project Structure

```text
zepto-data-ai-platform/
│
├── data_pipeline/
│   ├── data/
│   │   ├── books_clean.csv
│   │   └── zepto_books.db
│   ├── outputs/
│   │   ├── sql_query_outputs.txt
│   │   └── join_comparison.csv
│   └── scrape_pipeline.py
│
├── analytics/
│   ├── outputs/
│   ├── plots/
│   ├── 01_eda.py
│   ├── 02_modelling.py
│   └── titanic.csv
│
├── support_assistant/
│   ├── docs/
│   │   ├── doc_01.txt
│   │   ├── doc_02.txt
│   │   ├── doc_03.txt
│   │   ├── doc_04.txt
│   │   ├── doc_05.txt
│   │   ├── doc_06.txt
│   │   ├── doc_07.txt
│   │   └── doc_08.txt
│   ├── chroma_db/
│   ├── ingestion.py
│   ├── prompts.py
│   ├── retrieval.py
│   ├── graph.py
│   ├── schemas.py
│   ├── main.py
│   ├── Dockerfile
│   └── __init__.py
│
├── requirements.txt
└── README.md
1. Setup   
Python   
Python 3.11 is recommended.
Create a virtual environment:   

Bash
python -m venv .venv
Activate on Windows:   

PowerShell
.\.venv\Scripts\activate
Install dependencies:   

Bash
pip install -r requirements.txt
2. Module 1 — Data Pipeline
Objective
Scrape book information, clean the data, convert prices to INR, store the data in normalized SQLite tables, and demonstrate SQL and pandas analytics.   

Technologies   
Python   

Requests   

BeautifulSoup   

Pandas   

SQLite   

NumPy   

Run   
From the project root:   

PowerShell
python data_pipeline/scrape_pipeline.py
The pipeline scrapes the first five pages of Books to Scrape and collects at least 60 books across multiple categories.   

Fields collected
title

price

star rating

availability

category

Cleaning
The scraped price is converted into a numeric price_gbp.

Star ratings are converted from text to integers:

One -> 1

Two -> 2

Three -> 3

Four -> 4

Five -> 5

Availability is converted into a Boolean in_stock value.

Numeric parsing failures are handled using median imputation. Rows with essential text fields that cannot be parsed are removed.

Currency conversion
The project uses the fixed conversion rate:
1 GBP = INR 105.50

Therefore:
price_inr = price_gbp * 105.50

No external currency API is used.

SQLite design
The database contains normalized tables including:

categories

books

The categories table contains the category primary key and category name.

The books table contains book information and a foreign key referencing the category table.   

SQL analysis   
The pipeline demonstrates:   

SELECT and WHERE   

ORDER BY   

LIMIT   

DISTINCT   

BETWEEN   

JOIN   

SQL outputs are saved in:
data_pipeline/outputs/sql_query_outputs.txt   

Pandas JOIN comparison is saved in:
data_pipeline/outputs/join_comparison.csv   

The pipeline also reads SQL results using pandas.read_sql() and reproduces the JOIN using pandas.merge().   

3. Module 2 — Analytics and Machine Learning   
Part A — Exploratory Data Analysis   
Run:   

PowerShell
python analytics/01_eda.py
The Titanic dataset is loaded once using Seaborn and immediately saved as:
   analytics/titanic.csv

The saved CSV is used for subsequent analysis and modelling.

The analysis includes:

Dataset information

Descriptive statistics

Dataset shape

Missing-value analysis

Missing-value treatment

Age and fare distributions

Boxplots

IQR outlier analysis

Fare mean, median and mode

Survival rate analysis

Sex and passenger-class analysis

Correlation analysis

Correlation heatmap

Multivariate visualizations   

EDA-only standardization   

The correlation analysis uses exactly:   

survived   

pclass   

age   

sibsp   

parch   

fare   

Standardization of age and fare is performed only for EDA and is not used as a modelling preprocessing step.   

Part B — Machine Learning   
Run:   

PowerShell
python analytics/02_modelling.py
The modelling pipeline includes:   

Stratified train/test split

Train-only preprocessing

Missing-value handling

Categorical encoding

Numerical scaling

ColumnTransformer

Pipeline

Logistic Regression

Decision Tree

Random Forest

Confusion matrices

Accuracy

Precision

Recall

F1 score

ROC/AUC

Decision Tree visualization

Class imbalance comparison

Class-weight balancing

SMOTE

Random Forest GridSearchCV

OOB score

Regression side task

MAE

RMSE

R²

Adjusted R²

Residual analysis

Final fitted pipeline persistence using Joblib

The classification models use the same stratified train/test split for comparison.

The regression task predicts fare using other available features and evaluates the result separately from classification metrics.

4. Module 3 — Zepto Support Assistant
Objective
Build a local policy-based customer support assistant using:

Plaintext
Documents 
   ↓ 
Sentence Transformer Embeddings 
   ↓ 
ChromaDB 
   ↓ 
LangGraph 
   ↓ 
Policy Retrieval 
   ↓ 
Mock Answer 
   ↓ 
Pydantic Response 
   ↓ 
FastAPI
Technologies   
Python   

Sentence Transformers   

all-MiniLM-L6-v2   

ChromaDB   

LangGraph   

Pydantic   

FastAPI   

Uvicorn   

Document Ingestion   
The assistant uses eight Zepto policy documents stored in:
support_assistant/docs/   

Create the ChromaDB collection by running:   

PowerShell
python -m support_assistant.ingestion
The ingestion process:   

Reads the eight policy documents.

Loads the local all-MiniLM-L6-v2 embedding model.

Generates embeddings.

Stores the documents and embeddings in ChromaDB.

Tests retrieval using a sample policy question.

Retrieval
The retrieval module is:
support_assistant/retrieval.py

It loads the local embedding model and queries the ChromaDB collection using cosine-distance based retrieval.

The assistant retrieves the top three relevant policy documents.

Prompt Design
The prompt template contains:

Role

Context

Task

Format

Length

Negative constraint

Retrieved context

Customer question

A few-shot example is also included.

The negative constraint prevents the assistant from inventing policies, prices, timings or refund rules that are not present in the retrieved context.

5. LangGraph Workflow
The assistant uses a LangGraph StateGraph.

The state contains:

query

intent

retrieved_documents

answer

sources

confidence

The workflow contains three required nodes:

classify_intent

retrieve_and_answer

direct_answer

Flow:

Plaintext
┌──────────────────────┐
│   classify_intent    │
└──────────┬───────────┘
           │
 ┌─────────┴──────────┐
 │                    │
policy_question    general_question
 │                    │
 ▼                    ▼
┌────────────────────┐┌─────────────────┐
│ retrieve_and_answer││  direct_answer  │
└─────────┬──────────┘└────────┬────────┘
          │                    │
          └─────────┬──────────┘
                    ▼
                   END
Policy-related queries are retrieved from ChromaDB.   

General questions use the direct-answer branch.   

6. Mock LLM Mode   
The project uses a deterministic mock response path so that the baseline application does not require an external LLM API.   

For a policy question, the response follows the format:   

Based on the retrieved context: {top_chunk_snippet}   

For a general question:   

I can only answer questions about Zepto policies right now.   

This allows the application to run without network access to an external LLM provider.   

7. FastAPI   
Start the API from the project root:   

PowerShell
python -m uvicorn support_assistant.main:app
The API runs at:    http://127.0.0.1:8000

Swagger documentation: http://127.0.0.1:8000/docs

API Endpoint   
POST /ask
Request:   

JSON
{
  "query": "How long do I have to report damaged items?"
}
Example response:   

JSON
{
  "answer": "Based on the retrieved context: ...",
  "sources": [
    "doc_06.txt",
    "doc_02.txt",
    "doc_04.txt"
  ],
  "confidence": 1.0
}
General question query:   

JSON
{
  "query": "Tell me a joke."
}
Example response:   

JSON
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
The response is validated using Pydantic.   

8. Docker   
A Dockerfile is included at:
support_assistant/Dockerfile   

It provides a containerized deployment option for the FastAPI application.   

The application can also be run directly with Python and Uvicorn as described above.   

9. Design Decisions   
Local embeddings   
all-MiniLM-L6-v2 is used so embeddings can be generated locally without requiring a paid external embedding API.   

ChromaDB   
ChromaDB provides persistent local vector storage and similarity retrieval.   

LangGraph   
LangGraph separates intent classification, retrieval/answer generation and direct answering into explicit workflow nodes.   

Mock baseline   
The deterministic mock path allows the system to be demonstrated without requiring an external LLM API.   

Pydantic   
Pydantic validates the API response structure and ensures that confidence remains within the range 0 to 1.   

Fixed currency conversion   
The GBP-to-INR rate is fixed at 105.50. This ensures reproducible results without relying on an external exchange-rate service.   

10. Reproducibility   
The project can be regenerated from the source code using the provided scripts.   

Main execution order:   

PowerShell
python data_pipeline/scrape_pipeline.py
python analytics/01_eda.py
python analytics/02_modelling.py
python -m support_assistant.ingestion
python -m uvicorn support_assistant.main:app
11. Git Workflow
Development was performed using a feature branch before merging completed work into the main branch.

The repository contains the complete source code, analysis outputs, support documents and configuration required for the project.