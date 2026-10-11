"""SQLite access layer for SmartBiz AI.

The schema stores facts only (products, sales, sale lines, payments, expenses).
All financial figures are derived from these rows by SQL/Python in business_tools.py.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = "smartbiz.db"

_DATE_CHECK = "GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS products (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    sku           TEXT    NOT NULL UNIQUE,
    name          TEXT    NOT NULL,
    category      TEXT    NOT NULL DEFAULT 'General',
    cost_price    REAL    NOT NULL CHECK (cost_price >= 0),
    selling_price REAL    NOT NULL CHECK (selling_price >= 0),
    stock_qty     INTEGER NOT NULL DEFAULT 0 CHECK (stock_qty >= 0),
    reorder_level INTEGER NOT NULL DEFAULT 0 CHECK (reorder_level >= 0),
    is_active     INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
);

CREATE TABLE IF NOT EXISTS customers (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name  TEXT NOT NULL,
    phone TEXT
);

-- One row per sale (invoice). Totals are NOT stored: they are computed from sale_items.
-- customer_id NULL means a walk-in customer (assumed to have paid in full).
CREATE TABLE IF NOT EXISTS sales (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sale_date   TEXT NOT NULL CHECK (sale_date {_DATE_CHECK}),
    customer_id INTEGER REFERENCES customers(id),
    amount_paid REAL NOT NULL DEFAULT 0 CHECK (amount_paid >= 0),
    notes       TEXT,
    request_key TEXT
);

-- unit_cost is a snapshot of the product cost at the time of sale, so COGS stays
-- correct even if the product's cost price changes later.
CREATE TABLE IF NOT EXISTS sale_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    sale_id    INTEGER NOT NULL REFERENCES sales(id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    unit_price REAL    NOT NULL CHECK (unit_price >= 0),
    unit_cost  REAL    NOT NULL CHECK (unit_cost >= 0)
);

-- Money received from a customer after the sale (settling credit).
CREATE TABLE IF NOT EXISTS customer_payments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id  INTEGER NOT NULL REFERENCES customers(id),
    payment_date TEXT NOT NULL CHECK (payment_date {_DATE_CHECK}),
    amount       REAL NOT NULL CHECK (amount > 0),
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS expenses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    expense_date TEXT NOT NULL CHECK (expense_date {_DATE_CHECK}),
    category     TEXT NOT NULL,
    description  TEXT,
    amount       REAL NOT NULL CHECK (amount >= 0)
);

CREATE INDEX IF NOT EXISTS idx_sales_date        ON sales(sale_date);
CREATE INDEX IF NOT EXISTS idx_sales_customer    ON sales(customer_id);
CREATE INDEX IF NOT EXISTS idx_items_sale        ON sale_items(sale_id);
CREATE INDEX IF NOT EXISTS idx_items_product     ON sale_items(product_id);
CREATE INDEX IF NOT EXISTS idx_payments_customer ON customer_payments(customer_id);
CREATE INDEX IF NOT EXISTS idx_payments_date     ON customer_payments(payment_date);
CREATE INDEX IF NOT EXISTS idx_expenses_date     ON expenses(expense_date);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sales_request_key ON sales(request_key) WHERE request_key IS NOT NULL;
"""

TABLES_IN_DELETE_ORDER = (
    "sale_items",
    "customer_payments",
    "sales",
    "expenses",
    "customers",
    "products",
)


def load_env() -> None:
    """Load variables from a .env file if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is in requirements.txt
        return
    load_dotenv()


def get_db_path() -> str:
    return os.getenv("SMARTBIZ_DB", DEFAULT_DB_PATH)


def get_connection(db_path: str | None = None, read_only: bool = False) -> sqlite3.Connection:
    """Open a SQLite connection.

    read_only=True opens the file in SQLite's read-only mode AND sets
    PRAGMA query_only, so the AI agent physically cannot modify business records.
    """
    path = db_path or get_db_path()
    if read_only:
        uri = f"{Path(path).resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if read_only:
        conn.execute("PRAGMA query_only = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes if they do not exist yet."""
    legacy_sales = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sales'").fetchone()
    if legacy_sales:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(sales)")}
        if "request_key" not in columns:
            conn.execute("ALTER TABLE sales ADD COLUMN request_key TEXT")
    conn.executescript(SCHEMA)
    conn.commit()


def init_db_file(db_path: str | None = None) -> None:
    conn = get_connection(db_path)
    try:
        init_db(conn)
    finally:
        conn.close()
