# SmartBiz AI

SmartBiz AI is a local-first business management application built with **FastAPI, SQLite, and Ollama**. The
browser dashboard, customer and product workflows, reports, and AI chat use the same local business database.
Financial totals are calculated by parameterized SQL and Python; the AI cannot run SQL or change records.

## Features

- Live dashboard: sales, COGS, gross profit, profit after recorded expenses, expenses, customer dues, catalog size,
  stock alerts, daily sales trend, recent transactions, and data-derived insights.
- Local Ollama chat with native model tool calling, bounded multi-step execution, validated tool arguments, and
  clear model/connection errors.
- Product/inventory search and editing; sale entry with stock checks, cost snapshots, atomic stock deduction,
  and request-id idempotency; customer and payment entry; expense entry.
- Sales and customer history, dues aging, daily/weekly/month-to-date reports, and period comparisons.
- Responsive browser interface; protected APIs can use an optional local shared token.

## Requirements

- Windows 10/11, Python 3.10 or later.
- Ollama desktop/service with a model that supports tool calling. The default `qwen2.5:3b` is suitable for modest
  local setups; models use disk, memory, and CPU/GPU resources.

## Install Ollama and a model

1. Install Ollama from [ollama.com/download](https://ollama.com/download) and leave its local service running.
2. Open PowerShell and download the default model:

   ```powershell
   ollama pull qwen2.5:3b
   ```

   Check installed models using `ollama list`. To use another tool-calling model, set `OLLAMA_MODEL` in `.env`
   to its exact downloaded name.

Ollama serves its native API at `http://localhost:11434`. SmartBiz uses `/api/chat` and `/api/tags` on that local
service. No cloud LLM, OpenAI account, OpenAI API key, or OpenAI SDK is used.

## Install and configure SmartBiz

From PowerShell:

```powershell
cd C:\path\to\smartbiz-ai
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and configure:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_TIMEOUT_SECONDS=120
SMARTBIZ_DB=smartbiz.db
BUSINESS_CURRENCY=USD
SMARTBIZ_API_TOKEN=
```

`SMARTBIZ_API_TOKEN` is optional. If non-empty, the browser asks for it on the first API request and keeps it in
the current tab's session storage; API clients must send it in the `X-API-Key` header. Keep this app local unless
it is protected with proper authentication and HTTPS. The `.env` file is ignored by Git.

## Start

Make sure Ollama is running, then:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

- App: <http://127.0.0.1:8000/>
- API docs: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health>
- Ollama/model status: <http://127.0.0.1:8000/api/ollama/status>

## Business data

An empty database is initialized automatically. Add products, customers, sales, and expenses through the
dashboard. A sale validates available product stock, records the unit-cost snapshot, and deducts stock as one
SQLite transaction. Optional request UUIDs make repeat submissions idempotent. A customer payment cannot exceed
their outstanding balance.

To load the built-in sample dataset into an empty database:

```powershell
python seed_data.py
```

The seed script refuses to overwrite existing business data. `python seed_data.py --reset` explicitly replaces
it; this operation deletes existing business records.

### Financial definitions

- **Sales revenue**: quantity × recorded sale-line price, whether paid or not.
- **COGS**: quantity × sale-line unit-cost snapshot.
- **Gross profit**: sales revenue − COGS; operating expenses are not deducted.
- **Profit after recorded expenses**: gross profit − expenses stored for that period.
- **Customer dues**: billed customer sales − payments at sale − subsequent customer payments.
- **Recorded cash flow**: cash actually received during the period − recorded expenses. Unpaid credit sales,
  inventory purchases, loans, and owner withdrawals are not included.

## API highlights

Read endpoints include:

- `GET /api/dashboard?period=this_month`
- `GET /api/sales`, `GET /api/sales/summary?period=today`, `GET /api/sales/top-products`
- `GET /api/products`, `GET /api/inventory`, `GET /api/inventory/low-stock`
- `GET /api/customers`, `GET /api/customers/dues`
- `GET /api/expenses`, `GET /api/expenses/summary`
- `GET /api/reports/daily`, `/weekly`, or `/monthly`
- `GET /api/recommendations`, `GET /api/ollama/status`

Validated workflows include `POST/PUT /api/products`, `POST /api/sales`, `POST /api/customers`,
`POST /api/customers/{id}/payments`, and `POST /api/expenses`. There are no AI tools for writes, arbitrary SQL,
or code execution. Input validation and SQLite constraints apply to business workflows.

The agent's read-only tool registry includes sales summaries/by-date, product rankings, inventory and low-stock,
restock recommendations, customer dues, expense summaries, period comparisons, business reports, and
recommendations. Tool argument schemas are supplied to Ollama; execution independently validates tool names and
arguments.

## Test

With the project environment active:

```powershell
python -m unittest test_tools test_agent test_api
```

The unit suite uses fixed sample data and mocked Ollama HTTP responses, so it doesn't require a running model.
For a local integration smoke test, start Ollama and SmartBiz, then ask a question in the browser. The Settings
screen and `/api/ollama/status` report service/model availability.

## Troubleshooting

- **Ollama is not responding**: open the Ollama desktop app; confirm `http://localhost:11434/api/tags` responds.
- **Model missing**: run `ollama pull qwen2.5:3b`, or download the name shown by `OLLAMA_MODEL`.
- **Tool calls unsupported/rejected**: configure a model/version with native function/tool calling; verify the
  model name with `ollama list`.
- **Slow response**: first inference loads model weights into memory. Try a smaller tool-calling model or raise
  `OLLAMA_TIMEOUT_SECONDS`.
- **401 Missing or invalid X-API-Key**: leave `SMARTBIZ_API_TOKEN` blank for local-only use, or reload the app
  and enter the configured value when the browser prompts for it.
- **No sales shown**: verify the selected date period and confirm sales exist in the database. Empty records are
  explicitly distinguished from positive sales; the dashboard does not invent production statistics.

## Project structure

```text
main.py             FastAPI routes and validated business workflows
database.py         SQLite schema, safe legacy migration, connection access
business_tools.py   Parameterized read-only analytics and AI tool registry
agent.py            Ollama native tool-calling loop
schemas.py           Request validation models
static/index.html   Responsive dashboard, modules, and assistant
seed_data.py         Optional sample data
test_tools.py        Business calculation tests
test_agent.py        Tool-loop, Ollama adapter, and error-handling tests
test_api.py          API, dashboard, validation, and business workflow tests
```
