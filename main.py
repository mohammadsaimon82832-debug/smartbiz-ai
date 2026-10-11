"""SmartBiz AI - FastAPI application.

Run:  uvicorn main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import hmac
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import database

database.load_env()  # must run before the modules below read environment variables

import business_tools as bt  # noqa: E402
from agent import AgentError, SmartBizAgent, ai_is_configured  # noqa: E402
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse  # noqa: E402
from schemas import (  # noqa: E402
    DATE_PATTERN, ChatRequest, ChatResponse, CustomerWrite, ErrorResponse, ExpenseWrite, PaymentWrite,
    PeriodName, ProductWrite, ReportType, SaleWrite,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("smartbiz")

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db_file()  # creates the database file and tables if they do not exist
    if not ai_is_configured():
        logger.warning("AI provider is not configured: /api/chat is disabled, all other endpoints work.")
    else:
        logger.info("Ollama is selected; ensure the local Ollama service and configured model are available.")
    yield


app = FastAPI(
    title="SmartBiz AI",
    description="Read-only AI business management agent: sales, inventory, dues, expenses and reports.",
    version="1.0.0",
    lifespan=lifespan,
)

_agent: Optional[SmartBizAgent] = None


def get_agent() -> SmartBizAgent:
    global _agent
    if _agent is None:
        _agent = SmartBizAgent()
    return _agent


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
def require_token(x_api_key: Optional[str] = Header(default=None)) -> None:
    """If SMARTBIZ_API_TOKEN is set, require it in the X-API-Key header."""
    expected = os.getenv("SMARTBIZ_API_TOKEN", "")
    if expected and not hmac.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


def get_db():
    """One READ-ONLY connection per request: the API and the AI cannot change business records."""
    try:
        conn = database.get_connection(read_only=True)
    except sqlite3.Error:
        logger.exception("Could not open the database")
        raise HTTPException(status_code=503, detail="The business database is unavailable.")
    try:
        yield conn
    finally:
        conn.close()


def get_write_db():
    """Open a connection for validated application workflows (never passed to the AI)."""
    try:
        conn = database.get_connection()
    except sqlite3.Error:
        logger.exception("Could not open the database for a write")
        raise HTTPException(status_code=503, detail="The business database is unavailable.")
    try:
        yield conn
    finally:
        conn.close()


def run_tool(conn: sqlite3.Connection, name: str, **params: Any) -> dict:
    params = {k: v for k, v in params.items() if v is not None}
    result = bt.execute_tool(conn, name, params)
    if result.get("status") == "error":
        code = 422 if result.get("error_type") == "invalid_input" else 500
        raise HTTPException(status_code=code, detail=result.get("message", "Tool error."))
    return result


@app.exception_handler(sqlite3.Error)
async def sqlite_error_handler(request: Request, exc: sqlite3.Error):
    logger.error("Database error on %s: %s", request.url.path, type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "A database error occurred."})


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
api = APIRouter(prefix="/api", dependencies=[Depends(require_token)])

DateParam = Annotated[Optional[str], Query(pattern=DATE_PATTERN, description="YYYY-MM-DD")]


@api.get("/ollama/status")
def ollama_status():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
    try:
        import httpx
        response = httpx.get(f"{base_url}/api/tags", timeout=3)
        response.raise_for_status()
        models = response.json().get("models", [])
        installed = any(m.get("name") == model or m.get("name") == f"{model}:latest" for m in models)
        return {
            "connected": True,
            "model": model,
            "model_available": installed,
            "message": "Ready" if installed else f"Model not downloaded. Run: ollama pull {model}",
        }
    except ImportError:
        return {"connected": False, "model": model, "model_available": False,
                "message": "Install project dependencies to check Ollama status."}
    except Exception as exc:
        logger.info("Ollama status check failed (%s)", type(exc).__name__)
        return {"connected": False, "model": model, "model_available": False,
                "message": "Ollama is not responding. Start the Ollama app and try again."}


@api.get("/dashboard")
def dashboard(period: PeriodName = "this_month", conn: sqlite3.Connection = Depends(get_db)):
    summary = bt.get_sales_summary(conn, period=period)
    dues = bt.get_customer_dues(conn)
    inventory_data = bt.get_inventory_status(conn)
    low_stock = bt.get_low_stock_products(conn)
    recommendations_data = bt.get_business_recommendations(conn, period=period)
    start = (date.today() - timedelta(days=13)).isoformat()
    end = date.today().isoformat()
    trend_rows = conn.execute(
        """SELECT s.sale_date, ROUND(SUM(si.quantity * si.unit_price), 2) AS total
           FROM sales s JOIN sale_items si ON si.sale_id = s.id
           WHERE s.sale_date BETWEEN ? AND ? GROUP BY s.sale_date ORDER BY s.sale_date""",
        (start, end),
    ).fetchall()
    trend = {row["sale_date"]: row["total"] for row in trend_rows}
    daily_sales = [
        {"date": (date.today() - timedelta(days=offset)).isoformat(),
         "total_sales": float(trend.get((date.today() - timedelta(days=offset)).isoformat(), 0))}
        for offset in range(13, -1, -1)
    ]
    recent = conn.execute(
        """SELECT s.id, s.sale_date, s.customer_id, COALESCE(c.name, 'Walk-in') AS customer_name,
                  ROUND(COALESCE(SUM(si.quantity * si.unit_price), 0), 2) AS total,
                  s.amount_paid
           FROM sales s LEFT JOIN customers c ON c.id = s.customer_id
           LEFT JOIN sale_items si ON si.sale_id = s.id
           GROUP BY s.id ORDER BY s.sale_date DESC, s.id DESC LIMIT 8"""
    ).fetchall()
    active_products = conn.execute("SELECT COUNT(*) FROM products WHERE is_active=1").fetchone()[0]
    return {
        "currency": os.getenv("BUSINESS_CURRENCY", "USD"),
        "period": summary["period"],
        "kpis": {
            "total_sales": summary["total_sales"],
            "gross_profit": summary["gross_profit"],
            "profit_after_recorded_expenses": summary["profit_after_recorded_expenses"],
            "total_expenses": summary["total_recorded_expenses"],
            "outstanding_sales_balances": dues.get("total_outstanding", 0),
            "total_products": active_products,
            "low_stock_products": low_stock.get("low_stock_count", 0),
            "sale_count": summary["sale_count"],
        },
        "daily_sales": daily_sales,
        "recent_transactions": [dict(row) for row in recent],
        "inventory_alerts": low_stock.get("products", [])[:5],
        "insights": recommendations_data.get("insights", [])[:5],
        "data_status": summary.get("status", "ok"),
        "data_note": summary.get("message"),
        "inventory_status": inventory_data.get("status", "ok"),
    }


@api.post("/chat", response_model=ChatResponse,
          responses={429: {"model": ErrorResponse}, 502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def chat(body: ChatRequest, conn: sqlite3.Connection = Depends(get_db)):
    """Ask a business question in plain language. The AI picks tools and answers from real records."""
    try:
        result = get_agent().ask(body.question, conn, [m.model_dump() for m in body.history])
    except AgentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    return ChatResponse(answer=result.answer, tools_used=result.tools_used, model=result.model)


@api.get("/sales/summary")
def sales_summary(period: Optional[PeriodName] = None, start_date: DateParam = None,
                  end_date: DateParam = None, conn: sqlite3.Connection = Depends(get_db)):
    return run_tool(conn, "get_sales_summary", period=period, start_date=start_date, end_date=end_date)


@api.get("/sales")
def sales_list(start_date: DateParam = None, end_date: DateParam = None,
               search: Optional[str] = Query(None, max_length=100),
               limit: int = Query(50, ge=1, le=200), conn: sqlite3.Connection = Depends(get_db)):
    where, params = [], []
    if start_date:
        where.append("s.sale_date >= ?")
        params.append(start_date)
    if end_date:
        where.append("s.sale_date <= ?")
        params.append(end_date)
    if search and search.strip():
        term = f"%{search.strip()}%"
        where.append("(CAST(s.id AS TEXT) LIKE ? OR COALESCE(c.name, 'Walk-in') LIKE ? OR COALESCE(s.notes, '') LIKE ?)")
        params.extend([term, term, term])
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = conn.execute(
        f"""SELECT s.id, s.sale_date, s.customer_id, COALESCE(c.name, 'Walk-in') AS customer_name,
                   s.notes, s.amount_paid,
                   ROUND(COALESCE(SUM(si.quantity * si.unit_price), 0), 2) AS total,
                   COUNT(si.id) AS item_count
            FROM sales s LEFT JOIN customers c ON c.id = s.customer_id
            LEFT JOIN sale_items si ON si.sale_id = s.id
            {clause} GROUP BY s.id ORDER BY s.sale_date DESC, s.id DESC LIMIT ?""",
        (*params, limit),
    ).fetchall()
    return {"sales": [dict(row) for row in rows], "count": len(rows)}


@api.get("/sales/{sale_id:int}")
def sale_detail(sale_id: int, conn: sqlite3.Connection = Depends(get_db)):
    sale = conn.execute(
        """SELECT s.id, s.sale_date, s.customer_id, COALESCE(c.name, 'Walk-in') AS customer_name,
                  s.amount_paid, s.notes, ROUND(SUM(si.quantity * si.unit_price), 2) AS total
           FROM sales s LEFT JOIN customers c ON c.id=s.customer_id
           JOIN sale_items si ON si.sale_id=s.id WHERE s.id=? GROUP BY s.id""", (sale_id,),
    ).fetchone()
    if not sale:
        raise HTTPException(status_code=404, detail="Sale not found.")
    items = conn.execute(
        """SELECT p.id AS product_id, p.name, p.sku, si.quantity, si.unit_price, si.unit_cost,
                  ROUND(si.quantity * si.unit_price, 2) AS line_total
           FROM sale_items si JOIN products p ON p.id=si.product_id WHERE si.sale_id=? ORDER BY si.id""",
        (sale_id,),
    ).fetchall()
    return {"sale": dict(sale), "items": [dict(item) for item in items]}


@api.post("/sales", status_code=201)
def create_sale(body: SaleWrite, conn: sqlite3.Connection = Depends(get_write_db)):
    request_key = str(body.request_id) if body.request_id else None
    if request_key:
        previous = conn.execute("SELECT id FROM sales WHERE request_key=?", (request_key,)).fetchone()
        if previous:
            return {"sale_id": previous["id"], "duplicate": True}
    try:
        conn.execute("BEGIN IMMEDIATE")
        prepared = []
        total = 0.0
        for item in body.items:
            product = conn.execute(
                "SELECT id, name, cost_price, selling_price, stock_qty, is_active FROM products WHERE id=?",
                (item.product_id,),
            ).fetchone()
            if not product or not product["is_active"]:
                raise HTTPException(status_code=422, detail=f"Product {item.product_id} is unavailable.")
            if product["stock_qty"] < item.quantity:
                raise HTTPException(status_code=422, detail=f"Not enough stock for {product['name']}.")
            unit_price = item.unit_price if item.unit_price is not None else product["selling_price"]
            total += item.quantity * unit_price
            prepared.append((product, item.quantity, unit_price))
        total = round(total, 2)
        paid = total if body.customer_id is None and body.amount_paid is None else (body.amount_paid or 0)
        if body.customer_id is not None and not conn.execute(
                "SELECT 1 FROM customers WHERE id=?", (body.customer_id,)).fetchone():
            raise HTTPException(status_code=422, detail="Selected customer was not found.")
        if paid > total:
            raise HTTPException(status_code=422, detail="Amount paid cannot exceed the sale total.")
        sale_id = conn.execute(
            """INSERT INTO sales (sale_date, customer_id, amount_paid, notes, request_key)
               VALUES (?, ?, ?, ?, ?)""",
            (body.sale_date.isoformat(), body.customer_id, paid, body.notes, request_key),
        ).lastrowid
        for product, quantity, unit_price in prepared:
            conn.execute(
                """INSERT INTO sale_items (sale_id, product_id, quantity, unit_price, unit_cost)
                   VALUES (?, ?, ?, ?, ?)""",
                (sale_id, product["id"], quantity, unit_price, product["cost_price"]),
            )
            changed = conn.execute(
                "UPDATE products SET stock_qty=stock_qty-? WHERE id=? AND stock_qty>=?",
                (quantity, product["id"], quantity),
            ).rowcount
            if changed != 1:
                raise HTTPException(status_code=409, detail=f"Stock changed while recording {product['name']}.")
        conn.commit()
        return {"sale_id": sale_id, "total": total, "amount_paid": paid, "duplicate": False}
    except HTTPException:
        conn.rollback()
        raise
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        if request_key and conn.execute("SELECT id FROM sales WHERE request_key=?", (request_key,)).fetchone():
            return {"sale_id": conn.execute("SELECT id FROM sales WHERE request_key=?", (request_key,)).fetchone()["id"],
                    "duplicate": True}
        logger.info("Sale was rejected by database constraints (%s)", type(exc).__name__)
        raise HTTPException(status_code=422, detail="Sale could not be recorded because its data is invalid.") from exc
    except sqlite3.Error:
        conn.rollback()
        logger.exception("Could not record sale")
        raise HTTPException(status_code=500, detail="Sale could not be recorded.")


@api.get("/sales/top-products")
def top_products(period: Optional[PeriodName] = None, start_date: DateParam = None,
                 end_date: DateParam = None, limit: int = Query(5, ge=1, le=50),
                 sort_by: Literal["revenue", "profit", "units"] = "revenue",
                 conn: sqlite3.Connection = Depends(get_db)):
    return run_tool(conn, "get_top_products", period=period, start_date=start_date, end_date=end_date,
                    limit=limit, sort_by=sort_by)


@api.get("/products")
def products_list(search: Optional[str] = Query(None, max_length=100),
                  category: Optional[str] = Query(None, max_length=100),
                  include_inactive: bool = False, conn: sqlite3.Connection = Depends(get_db)):
    where, params = [], []
    if not include_inactive:
        where.append("is_active=1")
    if search and search.strip():
        term = f"%{search.strip()}%"
        where.append("(name LIKE ? OR sku LIKE ? OR category LIKE ?)")
        params.extend([term, term, term])
    if category and category.strip():
        where.append("LOWER(category)=LOWER(?)")
        params.append(category.strip())
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = conn.execute(
        f"""SELECT id, sku, name, category, cost_price, selling_price, stock_qty, reorder_level, is_active
            FROM products {clause} ORDER BY name""", params,
    ).fetchall()
    return {"products": [dict(row) for row in rows], "count": len(rows)}


@api.post("/products", status_code=201)
def create_product(body: ProductWrite, conn: sqlite3.Connection = Depends(get_write_db)):
    try:
        cur = conn.execute(
            """INSERT INTO products (sku, name, category, cost_price, selling_price, stock_qty, reorder_level, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (body.sku, body.name, body.category, body.cost_price, body.selling_price, body.stock_qty,
             body.reorder_level, int(body.is_active)),
        )
        conn.commit()
        return {"product_id": cur.lastrowid, "message": "Product added."}
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise HTTPException(status_code=409, detail="That SKU is already in use or product data is invalid.") from exc


@api.put("/products/{product_id}")
def update_product(product_id: int, body: ProductWrite, conn: sqlite3.Connection = Depends(get_write_db)):
    try:
        changed = conn.execute(
            """UPDATE products SET sku=?, name=?, category=?, cost_price=?, selling_price=?,
                      stock_qty=?, reorder_level=?, is_active=? WHERE id=?""",
            (body.sku, body.name, body.category, body.cost_price, body.selling_price, body.stock_qty,
             body.reorder_level, int(body.is_active), product_id),
        ).rowcount
        if not changed:
            conn.rollback()
            raise HTTPException(status_code=404, detail="Product not found.")
        conn.commit()
        return {"product_id": product_id, "message": "Product updated."}
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise HTTPException(status_code=409, detail="That SKU is already in use or product data is invalid.") from exc


@api.get("/inventory")
def inventory(category: Optional[str] = Query(None, max_length=100), include_inactive: bool = False,
              conn: sqlite3.Connection = Depends(get_db)):
    return run_tool(conn, "get_inventory_status", category=category, include_inactive=include_inactive)


@api.get("/inventory/low-stock")
def low_stock(conn: sqlite3.Connection = Depends(get_db)):
    return run_tool(conn, "get_low_stock_products")


@api.get("/inventory/restock")
def restock(limit: int = Query(10, ge=1, le=50), conn: sqlite3.Connection = Depends(get_db)):
    return run_tool(conn, "get_restock_recommendations", limit=limit)


@api.get("/customers/dues")
def customer_dues(customer_name: Optional[str] = Query(None, max_length=100), limit: int = Query(20, ge=1, le=100),
                  conn: sqlite3.Connection = Depends(get_db)):
    return run_tool(conn, "get_customer_dues", customer_name=customer_name, limit=limit)


@api.get("/customers")
def customers_list(search: Optional[str] = Query(None, max_length=100), conn: sqlite3.Connection = Depends(get_db)):
    where, params = "", []
    if search and search.strip():
        where = "WHERE name LIKE ? OR COALESCE(phone, '') LIKE ?"
        params = [f"%{search.strip()}%", f"%{search.strip()}%"]
    rows = conn.execute(f"SELECT id, name, phone FROM customers {where} ORDER BY name", params).fetchall()
    dues = bt.get_customer_dues(conn, limit=100)["customers"]
    balances = {customer["customer_id"]: customer["outstanding_balance"] for customer in dues}
    return {"customers": [{**dict(row), "outstanding_balance": balances.get(row["id"], 0.0)} for row in rows],
            "count": len(rows)}


@api.post("/customers", status_code=201)
def create_customer(body: CustomerWrite, conn: sqlite3.Connection = Depends(get_write_db)):
    cur = conn.execute("INSERT INTO customers (name, phone) VALUES (?, ?)",
                       (body.name, body.phone.strip() if body.phone else None))
    conn.commit()
    return {"customer_id": cur.lastrowid, "message": "Customer added."}


@api.post("/customers/{customer_id}/payments", status_code=201)
def record_customer_payment(customer_id: int, body: PaymentWrite,
                            conn: sqlite3.Connection = Depends(get_write_db)):
    conn.execute("BEGIN IMMEDIATE")
    if not conn.execute("SELECT 1 FROM customers WHERE id=?", (customer_id,)).fetchone():
        conn.rollback()
        raise HTTPException(status_code=404, detail="Customer not found.")
    customer_name = conn.execute("SELECT name FROM customers WHERE id=?", (customer_id,)).fetchone()["name"]
    dues = bt.get_customer_dues(conn, customer_name=customer_name, limit=100, today=body.payment_date)
    customer = next((c for c in dues.get("customers", []) if c["customer_id"] == customer_id), None)
    if not customer:
        conn.rollback()
        raise HTTPException(status_code=422, detail="This customer has no outstanding balance.")
    if body.amount > customer["outstanding_balance"]:
        conn.rollback()
        raise HTTPException(status_code=422, detail="Payment cannot exceed the customer's outstanding balance.")
    cur = conn.execute(
        "INSERT INTO customer_payments (customer_id, payment_date, amount, notes) VALUES (?, ?, ?, ?)",
        (customer_id, body.payment_date.isoformat(), body.amount, body.notes),
    )
    conn.commit()
    return {"payment_id": cur.lastrowid, "message": "Payment recorded."}


@api.get("/expenses")
def expenses_list(start_date: DateParam = None, end_date: DateParam = None,
                  search: Optional[str] = Query(None, max_length=100),
                  limit: int = Query(100, ge=1, le=500), conn: sqlite3.Connection = Depends(get_db)):
    where, params = [], []
    if start_date:
        where.append("expense_date>=?")
        params.append(start_date)
    if end_date:
        where.append("expense_date<=?")
        params.append(end_date)
    if search and search.strip():
        term = f"%{search.strip()}%"
        where.append("(category LIKE ? OR COALESCE(description,'') LIKE ?)")
        params.extend([term, term])
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = conn.execute(
        f"""SELECT id, expense_date, category, description, amount FROM expenses {clause}
            ORDER BY expense_date DESC, id DESC LIMIT ?""", (*params, limit),
    ).fetchall()
    return {"expenses": [dict(row) for row in rows], "count": len(rows)}


@api.post("/expenses", status_code=201)
def create_expense(body: ExpenseWrite, conn: sqlite3.Connection = Depends(get_write_db)):
    cur = conn.execute(
        "INSERT INTO expenses (expense_date, category, description, amount) VALUES (?, ?, ?, ?)",
        (body.expense_date.isoformat(), body.category, body.description, body.amount),
    )
    conn.commit()
    return {"expense_id": cur.lastrowid, "message": "Expense recorded."}


@api.get("/expenses/summary")
def expense_summary(period: Optional[PeriodName] = None, start_date: DateParam = None,
                    end_date: DateParam = None, top_n: int = Query(5, ge=1, le=50),
                    conn: sqlite3.Connection = Depends(get_db)):
    return run_tool(conn, "get_expense_summary", period=period, start_date=start_date, end_date=end_date,
                    top_n=top_n)


@api.get("/recommendations")
def recommendations(period: Optional[PeriodName] = None, start_date: DateParam = None,
                    end_date: DateParam = None, conn: sqlite3.Connection = Depends(get_db)):
    return run_tool(conn, "get_business_recommendations", period=period, start_date=start_date,
                    end_date=end_date)


@api.get("/reports/{report_type}")
def report(report_type: ReportType, report_date: Optional[str] = Query(None, pattern=DATE_PATTERN),
           format: Literal["json", "text"] = "json", conn: sqlite3.Connection = Depends(get_db)):
    """Daily, weekly (7 days) or monthly (month-to-date) report. Add ?format=text for a printable version."""
    data = run_tool(conn, "get_business_report", report_type=report_type, report_date=report_date)
    if format == "text":
        return PlainTextResponse(bt.report_to_text(data))
    return data


app.include_router(api)


@app.get("/health")
def health():
    """Liveness check; does not expose any business data or secrets."""
    return {"status": "ok", "ai_configured": ai_is_configured()}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")
