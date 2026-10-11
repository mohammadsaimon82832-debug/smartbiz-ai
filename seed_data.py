"""Load realistic sample data into the SmartBiz AI database.

    python seed_data.py            # fills an empty database
    python seed_data.py --reset    # wipes ALL existing business data first

Dates are generated relative to today (last 60 days) so questions like
"how much did I sell today?" always have data. The generator is deterministic
for a given --seed value and date, and it only inserts raw facts (sales lines,
payments, expenses) - every total and profit figure is calculated later by SQL.
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from datetime import date, timedelta

import database

DAYS_OF_HISTORY = 60

# sku, name, category, cost, price, current stock, reorder level, popularity weight
PRODUCTS = [
    ("CBL-USBC-1M", "USB-C Cable 1m", "Accessories", 1.80, 4.50, 14, 25, 10),
    ("CHG-20W", "20W Fast Charger", "Accessories", 6.00, 12.00, 38, 15, 6),
    ("EAR-BT-PRO", "Bluetooth Earbuds Pro", "Audio", 14.00, 29.00, 4, 10, 5),
    ("EAR-WIRED", "Wired Earphones", "Audio", 1.50, 4.00, 60, 20, 4),
    ("PWR-10K", "Power Bank 10000mAh", "Power", 9.50, 21.00, 0, 8, 4),
    ("PWR-20K", "Power Bank 20000mAh", "Power", 15.00, 33.00, 11, 6, 2),
    ("SPK-MINI", "Mini Bluetooth Speaker", "Audio", 11.00, 24.00, 9, 6, 2),
    ("CAS-PHN-A", "Phone Case (Model A)", "Cases", 1.20, 5.00, 85, 30, 5),
    ("CAS-PHN-B", "Phone Case (Model B)", "Cases", 1.20, 5.00, 120, 30, 0),  # never sells -> slow mover
    ("SCR-GLASS", "Tempered Glass Protector", "Cases", 0.60, 3.00, 70, 40, 7),
    ("MEM-64", "64GB Memory Card", "Storage", 5.50, 11.00, 22, 10, 3),
    ("HUB-4P", "4-Port USB Hub", "Accessories", 4.20, 10.00, 17, 8, 2),
]

CUSTOMERS = [
    ("Rahim Traders", "01700-000001"),
    ("Sultana Mobile Shop", "01700-000002"),
    ("Karim Electronics", "01700-000003"),
    ("Nasrin Gadgets", "01700-000004"),
    ("City Tech Corner", "01700-000005"),
    ("Abdul Store", "01700-000006"),
]

EXPENSE_RULES_MONTHLY = [
    # day of month, category, description, amount
    (1, "Rent", "Shop rent", 900.00),
    (5, "Utilities", "Electricity bill", 120.00),
    (5, "Utilities", "Internet", 35.00),
    (28, "Salaries", "Staff salaries", 1100.00),
]


def reset_database(conn: sqlite3.Connection) -> None:
    for table in database.TABLES_IN_DELETE_ORDER:
        conn.execute(f"DELETE FROM {table}")  # fixed table names from a constant, no user input
    conn.execute("DELETE FROM sqlite_sequence")
    conn.commit()


def seed_database(conn: sqlite3.Connection, today: date | None = None, seed: int = 42, reset: bool = False) -> dict:
    """Insert sample data. Refuses to touch a non-empty database unless reset=True."""
    today = today or date.today()
    database.init_db(conn)
    if reset:
        reset_database(conn)
    elif conn.execute("SELECT 1 FROM products LIMIT 1").fetchone() or conn.execute(
            "SELECT 1 FROM sales LIMIT 1").fetchone():
        raise RuntimeError("The database already contains data. Re-run with --reset to replace it.")

    rng = random.Random(seed)

    product_ids = {}
    weights = []
    for sku, name, category, cost, price, stock, reorder, weight in PRODUCTS:
        cur = conn.execute(
            """INSERT INTO products (sku, name, category, cost_price, selling_price, stock_qty, reorder_level)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sku, name, category, cost, price, stock, reorder),
        )
        product_ids[sku] = (cur.lastrowid, cost, price)
        weights.append(weight)
    skus = [p[0] for p in PRODUCTS]

    customer_ids = []
    for name, phone in CUSTOMERS:
        customer_ids.append(conn.execute("INSERT INTO customers (name, phone) VALUES (?, ?)", (name, phone)).lastrowid)

    n_sales = n_items = n_payments = 0
    credit_sales = []  # (customer_id, sale_date, due_amount)

    for offset in range(DAYS_OF_HISTORY - 1, -1, -1):
        day = today - timedelta(days=offset)
        count = rng.randint(2, 4) if day.weekday() >= 4 else rng.randint(3, 6)
        if offset == 0:
            count = max(count, 3)  # make sure "today" has activity
        for _ in range(count):
            customer_id = rng.choice(customer_ids) if rng.random() < 0.4 else None
            lines = []
            for sku in rng.choices(skus, weights=weights, k=rng.randint(1, 3)):
                pid, cost, price = product_ids[sku]
                qty = rng.randint(1, 3) if customer_id is None else rng.randint(3, 10)
                lines.append((pid, qty, price, cost))
            # merge duplicate products picked twice
            merged = {}
            for pid, qty, price, cost in lines:
                prev = merged.get(pid, (0, price, cost))
                merged[pid] = (prev[0] + qty, price, cost)
            total = sum(q * p for q, p, _ in merged.values())

            if customer_id is None or rng.random() < 0.5:
                paid = total
            else:
                paid = round(total * rng.choice([0.0, 0.3, 0.5, 0.7]), 2)
            sale_id = conn.execute(
                "INSERT INTO sales (sale_date, customer_id, amount_paid) VALUES (?, ?, ?)",
                (day.isoformat(), customer_id, paid),
            ).lastrowid
            n_sales += 1
            for pid, (qty, price, cost) in merged.items():
                conn.execute(
                    """INSERT INTO sale_items (sale_id, product_id, quantity, unit_price, unit_cost)
                       VALUES (?, ?, ?, ?, ?)""",
                    (sale_id, pid, qty, price, cost),
                )
                n_items += 1
            if customer_id is not None and total - paid > 0.005:
                credit_sales.append((customer_id, day, round(total - paid, 2)))

    # Customers settle about half of their older credit with later payments.
    for customer_id, sale_day, due in credit_sales:
        if (today - sale_day).days >= 7 and rng.random() < 0.5:
            amount = round(due * rng.choice([0.5, 1.0]), 2)
            pay_day = min(today, sale_day + timedelta(days=rng.randint(3, 20)))
            conn.execute(
                "INSERT INTO customer_payments (customer_id, payment_date, amount, notes) VALUES (?, ?, ?, ?)",
                (customer_id, pay_day.isoformat(), amount, "Payment received"),
            )
            n_payments += 1

    n_expenses = 0
    for offset in range(DAYS_OF_HISTORY - 1, -1, -1):
        day = today - timedelta(days=offset)
        for dom, category, description, amount in EXPENSE_RULES_MONTHLY:
            if day.day == dom:
                conn.execute(
                    "INSERT INTO expenses (expense_date, category, description, amount) VALUES (?, ?, ?, ?)",
                    (day.isoformat(), category, description, amount),
                )
                n_expenses += 1
        if rng.random() < 0.35:
            conn.execute(
                "INSERT INTO expenses (expense_date, category, description, amount) VALUES (?, ?, ?, ?)",
                (day.isoformat(), "Transport", "Delivery and courier", round(rng.uniform(8, 35), 2)),
            )
            n_expenses += 1
        if rng.random() < 0.08:
            conn.execute(
                "INSERT INTO expenses (expense_date, category, description, amount) VALUES (?, ?, ?, ?)",
                (day.isoformat(), "Marketing", "Social media ads", round(rng.uniform(40, 90), 2)),
            )
            n_expenses += 1

    conn.commit()
    return {"products": len(PRODUCTS), "customers": len(CUSTOMERS), "sales": n_sales, "sale_items": n_items,
            "customer_payments": n_payments, "expenses": n_expenses}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Seed the SmartBiz AI database with sample data.")
    parser.add_argument("--db", help="Database path (default: SMARTBIZ_DB env var or smartbiz.db)")
    parser.add_argument("--reset", action="store_true", help="Delete all existing business data first")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args(argv)

    database.load_env()
    conn = database.get_connection(args.db)
    try:
        counts = seed_database(conn, seed=args.seed, reset=args.reset)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print("Sample data loaded into", args.db or database.get_db_path())
    for table, n in counts.items():
        print(f"  {table}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
