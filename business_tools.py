"""Read-only business analytics tools for SmartBiz AI.

Every financial number the assistant reports comes from these functions.
They run parameterized SQL SELECT statements and do the arithmetic in Python,
so the language model never has to calculate (or guess) a figure.

Definitions used everywhere (also returned in tool results as `definitions`):

* total_sales      = SUM(quantity * unit_price) of items sold in the period
* COGS             = SUM(quantity * unit_cost) of those same items
* gross_profit     = total_sales - COGS                (expenses NOT deducted)
* profit_after_recorded_expenses = gross_profit - expenses recorded in the period
* net_recorded_cash_flow = cash actually received - recorded expenses
                           (credit sales that are still unpaid are NOT cash)
"""
from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration / thresholds (documented in README)
# --------------------------------------------------------------------------- #
PERIOD_NAMES = (
    "today",
    "yesterday",
    "this_week",
    "last_7_days",
    "this_month",
    "last_30_days",
    "last_month",
    "this_year",
    "all_time",
)
DEFAULT_PERIOD = "last_30_days"

VELOCITY_WINDOW_DAYS = 30   # sales history used to estimate demand
TARGET_COVER_DAYS = 30      # restock up to this many days of expected sales
URGENT_COVER_DAYS = 3       # <= this many days of stock left -> critical
SOON_COVER_DAYS = 7         # <= this many days of stock left -> high

REVENUE_CHANGE_ALERT_PCT = 10.0     # revenue change worth flagging
REVENUE_DROP_HIGH_PCT = 25.0        # drop at/above this is "high" severity
LOW_MARGIN_PCT = 10.0               # gross margin below this is flagged
MARGIN_DROP_POINTS = 3.0            # margin fall (percentage points) vs previous
EXPENSE_TO_GROSS_PROFIT_WARN = 0.5  # expenses above 50% of gross profit
EXPENSE_CATEGORY_JUMP_PCT = 30.0    # category up 30%+ vs previous period
DUES_TO_SALES_WARN_PCT = 20.0       # outstanding dues vs period sales
DUES_TO_SALES_HIGH_PCT = 50.0
OVERDUE_DAYS = 60                   # dues older than this are "overdue"

DEFINITIONS = {
    "total_sales": "Sum of quantity x unit price of items sold in the period (revenue), "
                   "whether or not the customer has paid yet.",
    "cost_of_goods_sold": "Sum of quantity x unit cost recorded on each sold item at the time of sale.",
    "gross_profit": "total_sales minus cost_of_goods_sold. Operating expenses are NOT deducted.",
    "profit_after_recorded_expenses": "gross_profit minus the expenses recorded for the period. "
                                      "Only expenses present in the database are included.",
    "net_recorded_cash_flow": "Cash received (payments taken at the time of sale plus later customer "
                              "payments dated in the period) minus recorded expenses. Unpaid credit sales "
                              "are not cash. Inventory purchases, loans and owner withdrawals are not "
                              "recorded, so they are not included.",
}


class ToolInputError(ValueError):
    """Raised when a tool receives invalid arguments."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _money(value: Any) -> float:
    return round(float(value or 0), 2) + 0.0  # "+ 0.0" turns -0.0 into 0.0


def _pct(numerator: float, denominator: float) -> float | None:
    if not denominator:
        return None
    return round(numerator / denominator * 100, 2) + 0.0


def _pct_change(current: float, previous: float) -> float | None:
    if not previous:
        return None
    return round((current - previous) / abs(previous) * 100, 2) + 0.0


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def parse_date(value: Any, field: str = "date") -> date:
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not _DATE_RE.match(value.strip()):
        raise ToolInputError(f"{field} must be a date in YYYY-MM-DD format.")
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise ToolInputError(f"{field} '{value}' is not a valid calendar date.") from None


def _int_arg(value: Any, name: str, default: int, lo: int, hi: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolInputError(f"{name} must be an integer between {lo} and {hi}.")
    if not lo <= value <= hi:
        raise ToolInputError(f"{name} must be between {lo} and {hi}.")
    return value


def _opt_str(value: Any, name: str, max_len: int = 100) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolInputError(f"{name} must be text.")
    value = value.strip()
    if not value:
        return None
    if len(value) > max_len:
        raise ToolInputError(f"{name} is too long (max {max_len} characters).")
    return value


# --------------------------------------------------------------------------- #
# Periods
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Period:
    label: str
    start: date
    end: date

    @property
    def days(self) -> int | None:
        if self.label == "all_time":
            return None
        return (self.end - self.start).days + 1

    @property
    def start_iso(self) -> str:
        return self.start.isoformat()

    @property
    def end_iso(self) -> str:
        return self.end.isoformat()

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "start_date": None if self.label == "all_time" else self.start_iso,
            "end_date": self.end_iso,
            "days": self.days,
        }


def resolve_period(
    period: str | None = None,
    start_date: Any = None,
    end_date: Any = None,
    today: date | None = None,
) -> Period:
    """Turn a named period or explicit dates into an inclusive date range."""
    today = today or date.today()

    if start_date or end_date:
        if not start_date:
            raise ToolInputError("start_date is required when end_date is given.")
        start = parse_date(start_date, "start_date")
        end = parse_date(end_date, "end_date") if end_date else today
        if end < start:
            raise ToolInputError("end_date must not be before start_date.")
        return Period("custom", start, end)

    name = period or DEFAULT_PERIOD
    if not isinstance(name, str) or name not in PERIOD_NAMES:
        raise ToolInputError(f"Unknown period '{name}'. Valid periods: {', '.join(PERIOD_NAMES)}.")

    if name == "today":
        return Period(name, today, today)
    if name == "yesterday":
        y = today - timedelta(days=1)
        return Period(name, y, y)
    if name == "this_week":  # Monday .. today
        return Period(name, today - timedelta(days=today.weekday()), today)
    if name == "last_7_days":
        return Period(name, today - timedelta(days=6), today)
    if name == "this_month":
        return Period(name, today.replace(day=1), today)
    if name == "last_30_days":
        return Period(name, today - timedelta(days=29), today)
    if name == "last_month":
        end = today.replace(day=1) - timedelta(days=1)
        return Period(name, end.replace(day=1), end)
    if name == "this_year":
        return Period(name, today.replace(month=1, day=1), today)
    return Period("all_time", date.min, today)


def previous_period(p: Period) -> Period | None:
    """The window of equal length immediately before `p` (None for all_time)."""
    if p.label == "all_time" or p.days is None:
        return None
    end = p.start - timedelta(days=1)
    start = end - timedelta(days=p.days - 1)
    return Period("previous_period", start, end)


# --------------------------------------------------------------------------- #
# Core financial queries
# --------------------------------------------------------------------------- #
def _sales_totals(conn: sqlite3.Connection, start: str, end: str) -> dict:
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT s.id)                            AS sale_count,
               COALESCE(SUM(si.quantity), 0)                   AS units,
               COALESCE(SUM(si.quantity * si.unit_price), 0.0) AS revenue,
               COALESCE(SUM(si.quantity * si.unit_cost), 0.0)  AS cogs
        FROM sales s
        JOIN sale_items si ON si.sale_id = s.id
        WHERE s.sale_date BETWEEN ? AND ?
        """,
        (start, end),
    ).fetchone()
    return {"sale_count": row["sale_count"], "units": row["units"],
            "revenue": row["revenue"], "cogs": row["cogs"]}


def _expense_total(conn: sqlite3.Connection, start: str, end: str) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(amount), 0.0) AS total FROM expenses WHERE expense_date BETWEEN ? AND ?",
        (start, end),
    ).fetchone()
    return row["total"]


def _cash_received(conn: sqlite3.Connection, start: str, end: str) -> tuple[float, float]:
    at_sale = conn.execute(
        "SELECT COALESCE(SUM(amount_paid), 0.0) AS total FROM sales WHERE sale_date BETWEEN ? AND ?",
        (start, end),
    ).fetchone()["total"]
    later = conn.execute(
        "SELECT COALESCE(SUM(amount), 0.0) AS total FROM customer_payments WHERE payment_date BETWEEN ? AND ?",
        (start, end),
    ).fetchone()["total"]
    return at_sale, later


def _financials(conn: sqlite3.Connection, start: str, end: str) -> dict:
    """All headline numbers for one date range (rounded to 2 decimals)."""
    t = _sales_totals(conn, start, end)
    expenses = _expense_total(conn, start, end)
    at_sale, later = _cash_received(conn, start, end)
    gross = t["revenue"] - t["cogs"]
    after = gross - expenses
    cash_in = at_sale + later
    return {
        "sale_count": t["sale_count"],
        "units_sold": t["units"],
        "total_sales": _money(t["revenue"]),
        "cost_of_goods_sold": _money(t["cogs"]),
        "gross_profit": _money(gross),
        "gross_margin_pct": _pct(gross, t["revenue"]),
        "total_recorded_expenses": _money(expenses),
        "profit_after_recorded_expenses": _money(after),
        "net_margin_pct": _pct(after, t["revenue"]),
        "cash_collected_on_sales": _money(at_sale),
        "customer_payments_received": _money(later),
        "total_cash_received": _money(cash_in),
        "net_recorded_cash_flow": _money(cash_in - expenses),
    }


def _is_database_empty(conn: sqlite3.Connection) -> bool:
    for table in ("products", "sales", "expenses"):
        if conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():  # fixed table names, no user input
            return False
    return True


def _daily_breakdown(conn: sqlite3.Connection, start: str, end: str) -> list[dict]:
    days: dict[str, dict] = {}
    for r in conn.execute(
        """
        SELECT s.sale_date AS day,
               COUNT(DISTINCT s.id)                           AS n,
               SUM(si.quantity * si.unit_price)               AS revenue,
               SUM(si.quantity * si.unit_cost)                AS cogs
        FROM sales s JOIN sale_items si ON si.sale_id = s.id
        WHERE s.sale_date BETWEEN ? AND ?
        GROUP BY s.sale_date
        """,
        (start, end),
    ):
        days[r["day"]] = {"date": r["day"], "sale_count": r["n"], "total_sales": r["revenue"],
                          "cost_of_goods_sold": r["cogs"], "expenses": 0.0}
    for r in conn.execute(
        """SELECT expense_date AS day, SUM(amount) AS total FROM expenses
           WHERE expense_date BETWEEN ? AND ? GROUP BY expense_date""",
        (start, end),
    ):
        d = days.setdefault(r["day"], {"date": r["day"], "sale_count": 0, "total_sales": 0.0,
                                       "cost_of_goods_sold": 0.0, "expenses": 0.0})
        d["expenses"] = r["total"]
    out = []
    for day in sorted(days):
        d = days[day]
        gross = d["total_sales"] - d["cost_of_goods_sold"]
        out.append({
            "date": day,
            "sale_count": d["sale_count"],
            "total_sales": _money(d["total_sales"]),
            "cost_of_goods_sold": _money(d["cost_of_goods_sold"]),
            "gross_profit": _money(gross),
            "expenses": _money(d["expenses"]),
            "profit_after_recorded_expenses": _money(gross - d["expenses"]),
        })
    return out


# --------------------------------------------------------------------------- #
# Tool 1: sales summary
# --------------------------------------------------------------------------- #
def get_sales_summary(conn, period=None, start_date=None, end_date=None, today=None) -> dict:
    """Total sales, COGS, gross profit, profit after recorded expenses and cash flow."""
    p = resolve_period(period, start_date, end_date, today)
    result = {"status": "ok", "period": p.to_dict()}
    result.update(_financials(conn, p.start_iso, p.end_iso))
    result["definitions"] = DEFINITIONS
    result["notes"] = []
    if result["sale_count"] == 0:
        result["status"] = "no_sales"
        result["message"] = "No sales are recorded in this period."
        result["notes"].append("Figures are zero because there are no recorded sales in the period.")
    if result["total_recorded_expenses"] == 0:
        result["notes"].append("No expenses are recorded in this period, so profit after recorded "
                               "expenses equals gross profit.")
    if p.days is not None and p.days <= 31:
        result["daily_breakdown"] = _daily_breakdown(conn, p.start_iso, p.end_iso)
    return result


# --------------------------------------------------------------------------- #
# Tool 2: top products
# --------------------------------------------------------------------------- #
SORT_OPTIONS = ("revenue", "profit", "units")


def get_top_products(conn, period=None, start_date=None, end_date=None, limit=5,
                     sort_by="revenue", today=None) -> dict:
    p = resolve_period(period, start_date, end_date, today)
    limit = _int_arg(limit, "limit", 5, 1, 50)
    sort_by = sort_by or "revenue"
    if sort_by not in SORT_OPTIONS:
        raise ToolInputError(f"sort_by must be one of: {', '.join(SORT_OPTIONS)}.")

    rows = conn.execute(
        """
        SELECT p.id, p.sku, p.name,
               SUM(si.quantity)                  AS units,
               SUM(si.quantity * si.unit_price)  AS revenue,
               SUM(si.quantity * si.unit_cost)   AS cogs
        FROM sale_items si
        JOIN sales s    ON s.id = si.sale_id
        JOIN products p ON p.id = si.product_id
        WHERE s.sale_date BETWEEN ? AND ?
        GROUP BY p.id
        """,
        (p.start_iso, p.end_iso),
    ).fetchall()

    items = []
    for r in rows:
        gross = r["revenue"] - r["cogs"]
        items.append({
            "product_id": r["id"], "sku": r["sku"], "name": r["name"],
            "units_sold": r["units"],
            "total_sales": _money(r["revenue"]),
            "cost_of_goods_sold": _money(r["cogs"]),
            "gross_profit": _money(gross),
            "gross_margin_pct": _pct(gross, r["revenue"]),
        })
    key = {"revenue": "total_sales", "profit": "gross_profit", "units": "units_sold"}[sort_by]
    items.sort(key=lambda i: (-i[key], -i["total_sales"], i["name"]))

    result = {"status": "ok", "period": p.to_dict(), "sorted_by": sort_by,
              "products_with_sales": len(items), "products": items[:limit]}
    if not items:
        result["status"] = "no_sales"
        result["message"] = "No product sales are recorded in this period."
    return result


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
def get_inventory_status(conn, category=None, include_inactive=False, today=None) -> dict:
    category = _opt_str(category, "category")
    if include_inactive is None:
        include_inactive = False
    if not isinstance(include_inactive, bool):
        raise ToolInputError("include_inactive must be true or false.")

    sql = ("SELECT id, sku, name, category, cost_price, selling_price, stock_qty, reorder_level, is_active "
           "FROM products WHERE 1 = 1")
    params: list = []
    if not include_inactive:
        sql += " AND is_active = 1"
    if category:
        sql += " AND LOWER(category) = LOWER(?)"
        params.append(category)
    sql += " ORDER BY name"
    rows = conn.execute(sql, params).fetchall()

    if not rows:
        return {"status": "no_data", "message": "No products found" +
                (f" in category '{category}'." if category else "."),
                "product_count": 0, "products": []}

    items, cost_value, retail_value = [], 0.0, 0.0
    for r in rows:
        status = "out_of_stock" if r["stock_qty"] == 0 else (
            "low_stock" if r["stock_qty"] <= r["reorder_level"] else "ok")
        cost_value += r["stock_qty"] * r["cost_price"]
        retail_value += r["stock_qty"] * r["selling_price"]
        items.append({
            "product_id": r["id"], "sku": r["sku"], "name": r["name"], "category": r["category"],
            "stock_qty": r["stock_qty"], "reorder_level": r["reorder_level"], "stock_status": status,
            "cost_price": _money(r["cost_price"]), "selling_price": _money(r["selling_price"]),
            "stock_value_at_cost": _money(r["stock_qty"] * r["cost_price"]),
            "is_active": bool(r["is_active"]),
        })
    return {
        "status": "ok",
        "product_count": len(items),
        "out_of_stock_count": sum(1 for i in items if i["stock_status"] == "out_of_stock"),
        "low_stock_count": sum(1 for i in items if i["stock_status"] == "low_stock"),
        "total_stock_value_at_cost": _money(cost_value),
        "total_stock_value_at_selling_price": _money(retail_value),
        "products": items,
    }


_PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _classify_stock(stock: int, reorder: int, units_30d: int) -> tuple[str | None, str, float | None]:
    """Return (priority, reason, days_of_cover)."""
    is_low = stock <= reorder
    cover = None if units_30d == 0 else stock * VELOCITY_WINDOW_DAYS / units_30d

    if stock == 0 and units_30d > 0:
        return "critical", f"Out of stock; {units_30d} units sold in the last {VELOCITY_WINDOW_DAYS} days.", cover
    if cover is not None and cover <= URGENT_COVER_DAYS:
        return "critical", f"Only about {cover:.1f} days of stock left at the recent sales rate.", cover
    if cover is not None and cover <= SOON_COVER_DAYS:
        return "high", f"About {cover:.1f} days of stock left at the recent sales rate.", cover
    if is_low and units_30d > 0:
        return "medium", (f"Stock ({stock}) is at or below the reorder level ({reorder}); "
                          f"{units_30d} units sold in the last {VELOCITY_WINDOW_DAYS} days."), cover
    if is_low:
        return "low", (f"Stock ({stock}) is at or below the reorder level ({reorder}) "
                       f"but there were no sales in the last {VELOCITY_WINDOW_DAYS} days."), cover
    return None, "", cover


def _stock_analysis(conn: sqlite3.Connection, today: date) -> list[dict]:
    """Per-product stock position + restock suggestion for every active product."""
    start = (today - timedelta(days=VELOCITY_WINDOW_DAYS - 1)).isoformat()
    sold = {
        r["product_id"]: r["units"]
        for r in conn.execute(
            """
            SELECT si.product_id AS product_id, SUM(si.quantity) AS units
            FROM sale_items si JOIN sales s ON s.id = si.sale_id
            WHERE s.sale_date BETWEEN ? AND ?
            GROUP BY si.product_id
            """,
            (start, today.isoformat()),
        )
    }
    out = []
    for r in conn.execute(
        """SELECT id, sku, name, category, cost_price, stock_qty, reorder_level
           FROM products WHERE is_active = 1 ORDER BY name"""
    ):
        units = sold.get(r["id"], 0)
        priority, reason, cover = _classify_stock(r["stock_qty"], r["reorder_level"], units)
        qty = 0
        if priority is not None:
            target = max(_ceil_div(units * TARGET_COVER_DAYS, VELOCITY_WINDOW_DAYS), r["reorder_level"] * 2)
            qty = max(0, target - r["stock_qty"])
        out.append({
            "product_id": r["id"], "sku": r["sku"], "name": r["name"], "category": r["category"],
            "stock_qty": r["stock_qty"], "reorder_level": r["reorder_level"],
            "cost_price": _money(r["cost_price"]),
            "units_sold_last_30_days": units,
            "avg_daily_sales": round(units / VELOCITY_WINDOW_DAYS, 2),
            "days_of_cover": None if cover is None else round(cover, 1),
            "is_low_stock": r["stock_qty"] <= r["reorder_level"],
            "priority": priority,
            "reason": reason,
            "suggested_reorder_qty": qty,
            "estimated_restock_cost": _money(qty * r["cost_price"]),
        })
    return out


def _priority_sort_key(item: dict):
    cover = item["days_of_cover"]
    return (_PRIORITY_RANK[item["priority"]], cover if cover is not None else float("inf"), item["name"])


def _restock_view(item: dict) -> dict:
    return {k: item[k] for k in (
        "product_id", "sku", "name", "category", "stock_qty", "reorder_level", "units_sold_last_30_days",
        "days_of_cover", "priority", "reason", "suggested_reorder_qty", "estimated_restock_cost")}


RESTOCK_NOTE = ("Suggested quantity restocks up to the larger of (a) %d days of recent sales or (b) twice the "
                "reorder level. Supplier lead times and minimum order quantities are not recorded, so treat "
                "this as a starting point." % TARGET_COVER_DAYS)


def get_low_stock_products(conn, today=None) -> dict:
    """Active products at or below their reorder level, most urgent first."""
    today = today or date.today()
    analysis = _stock_analysis(conn, today)
    if not analysis:
        return {"status": "no_data", "message": "No active products are recorded.", "low_stock_count": 0,
                "products": []}
    low = sorted((a for a in analysis if a["is_low_stock"]), key=_priority_sort_key)
    result = {
        "status": "ok",
        "active_product_count": len(analysis),
        "low_stock_count": len(low),
        "out_of_stock_count": sum(1 for a in low if a["stock_qty"] == 0),
        "products": [_restock_view(a) for a in low],
        "note": RESTOCK_NOTE,
    }
    if not low:
        result["message"] = "No products are at or below their reorder level."
    return result


def get_restock_recommendations(conn, limit=10, today=None) -> dict:
    """Products to consider restocking: low stock, plus anything that will sell out within a week."""
    today = today or date.today()
    limit = _int_arg(limit, "limit", 10, 1, 50)
    analysis = _stock_analysis(conn, today)
    if not analysis:
        return {"status": "no_data", "message": "No active products are recorded.", "recommendations": []}
    candidates = sorted((a for a in analysis if a["priority"] is not None), key=_priority_sort_key)
    result = {
        "status": "ok",
        "candidate_count": len(candidates),
        "total_estimated_restock_cost_all_candidates": _money(sum(c["estimated_restock_cost"] for c in candidates)),
        "recommendations": [_restock_view(c) for c in candidates[:limit]],
        "priority_meaning": {
            "critical": "Out of stock with demand, or about 3 days of stock or less.",
            "high": "About 7 days of stock or less at the recent sales rate.",
            "medium": "At/below reorder level and selling.",
            "low": "At/below reorder level but no recent sales.",
        },
        "note": RESTOCK_NOTE,
    }
    if not candidates:
        result["message"] = "No products currently need restocking."
    return result


# --------------------------------------------------------------------------- #
# Customer dues
# --------------------------------------------------------------------------- #
AGING_BUCKETS = ("0_30_days", "31_60_days", "61_90_days", "over_90_days")


def _bucket(age_days: int) -> str:
    if age_days <= 30:
        return AGING_BUCKETS[0]
    if age_days <= 60:
        return AGING_BUCKETS[1]
    if age_days <= 90:
        return AGING_BUCKETS[2]
    return AGING_BUCKETS[3]


def _compute_dues(conn: sqlite3.Connection, today: date) -> dict:
    """Outstanding balance per customer.

    balance = billed - paid at sale - later payments  (negative = customer credit).
    Aging allocates later payments to the oldest unpaid sales first (FIFO).
    """
    customers = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM customers")}
    sales_by_customer: dict[int, list] = {}
    for r in conn.execute(
        """
        SELECT s.id, s.customer_id, s.sale_date, s.amount_paid,
               COALESCE(SUM(si.quantity * si.unit_price), 0.0) AS total
        FROM sales s LEFT JOIN sale_items si ON si.sale_id = s.id
        WHERE s.customer_id IS NOT NULL
        GROUP BY s.id
        ORDER BY s.sale_date, s.id
        """
    ):
        sales_by_customer.setdefault(r["customer_id"], []).append(r)
    payments = {
        r["customer_id"]: (r["total"], r["last_date"])
        for r in conn.execute(
            """SELECT customer_id, SUM(amount) AS total, MAX(payment_date) AS last_date
               FROM customer_payments GROUP BY customer_id"""
        )
    }

    records = []
    aging = {b: 0.0 for b in AGING_BUCKETS}
    for cid, name in customers.items():
        sales = sales_by_customer.get(cid, [])
        billed = sum(s["total"] for s in sales)
        paid_at_sale = sum(s["amount_paid"] for s in sales)
        later, last_payment = payments.get(cid, (0.0, None))
        balance = billed - paid_at_sale - later

        pool = later
        oldest = None
        customer_aging = {b: 0.0 for b in AGING_BUCKETS}
        for s in sales:
            unpaid = max(0.0, s["total"] - s["amount_paid"])
            applied = min(unpaid, pool)
            unpaid -= applied
            pool -= applied
            if unpaid > 0.005:
                if oldest is None:
                    oldest = s["sale_date"]
                age = max(0, (today - date.fromisoformat(s["sale_date"])).days)
                customer_aging[_bucket(age)] += unpaid
        if balance > 0.005:
            for b in AGING_BUCKETS:
                aging[b] += customer_aging[b]
        records.append({
            "customer_id": cid, "name": name,
            "outstanding_balance": _money(balance),
            "total_billed": _money(billed),
            "paid_at_time_of_sale": _money(paid_at_sale),
            "later_payments_received": _money(later),
            "last_payment_date": last_payment,
            "oldest_unpaid_sale_date": oldest if balance > 0.005 else None,
            "days_since_oldest_unpaid_sale": (max(0, (today - date.fromisoformat(oldest)).days)
                                              if oldest and balance > 0.005 else None),
            "_raw_balance": balance,
            "_over_60": customer_aging[AGING_BUCKETS[2]] + customer_aging[AGING_BUCKETS[3]],
        })
    return {"records": records, "aging": aging}


def get_customer_dues(conn, customer_name=None, limit=20, today=None) -> dict:
    """Outstanding customer balances (optionally for one customer, matched by partial name)."""
    today = today or date.today()
    name = _opt_str(customer_name, "customer_name")
    limit = _int_arg(limit, "limit", 20, 1, 100)

    data = _compute_dues(conn, today)
    records = data["records"]
    if not records:
        return {"status": "no_data", "message": "No customers are recorded.", "customers": []}

    def public(rec):
        return {k: v for k, v in rec.items() if not k.startswith("_")}

    if name:
        matches = [r for r in records if name.lower() in r["name"].lower()]
        if not matches:
            return {"status": "not_found", "message": f"No customer matching '{name}' was found.",
                    "customers": []}
        return {"status": "ok", "customer_filter": name,
                "customers": [public(r) for r in sorted(matches, key=lambda r: -r["_raw_balance"])[:limit]],
                "note": "A positive outstanding_balance means the customer owes money; negative means "
                        "the customer has paid in advance (credit)."}

    owing = sorted((r for r in records if r["_raw_balance"] > 0.005), key=lambda r: (-r["_raw_balance"], r["name"]))
    total = sum(r["_raw_balance"] for r in owing)
    result = {
        "status": "ok",
        "total_outstanding": _money(total),
        "customers_owing_count": len(owing),
        "customers_with_credit_count": sum(1 for r in records if r["_raw_balance"] < -0.005),
        "showing": min(limit, len(owing)),
        "customers": [public(r) for r in owing[:limit]],
        "aging_of_outstanding_by_sale_age": {b: _money(v) for b, v in data["aging"].items()},
        "note": "Aging buckets are by the age of each unpaid sale, after applying later payments to the "
                "oldest sales first. Walk-in sales (no customer) are assumed fully paid.",
    }
    if not owing:
        result["message"] = "No customers currently owe money."
    return result


# --------------------------------------------------------------------------- #
# Expenses
# --------------------------------------------------------------------------- #
def _expense_by_category(conn: sqlite3.Connection, start: str, end: str) -> dict[str, float]:
    return {
        r["category"]: r["total"]
        for r in conn.execute(
            """SELECT category, SUM(amount) AS total FROM expenses
               WHERE expense_date BETWEEN ? AND ? GROUP BY category""",
            (start, end),
        )
    }


def get_expense_summary(conn, period=None, start_date=None, end_date=None, top_n=5, today=None) -> dict:
    p = resolve_period(period, start_date, end_date, today)
    top_n = _int_arg(top_n, "top_n", 5, 1, 50)
    fin = _financials(conn, p.start_iso, p.end_iso)
    cats = _expense_by_category(conn, p.start_iso, p.end_iso)
    total = sum(cats.values())

    result = {"status": "ok", "period": p.to_dict()}
    if not cats:
        result.update({
            "status": "no_expenses",
            "message": "No expenses are recorded in this period.",
            "total_recorded_expenses": 0.0,
            "total_sales": fin["total_sales"],
            "gross_profit": fin["gross_profit"],
            "profit_after_recorded_expenses": fin["gross_profit"],
        })
        return result

    by_cat = sorted(cats.items(), key=lambda kv: (-kv[1], kv[0]))
    largest = conn.execute(
        """SELECT expense_date, category, description, amount FROM expenses
           WHERE expense_date BETWEEN ? AND ? ORDER BY amount DESC, expense_date DESC, id LIMIT ?""",
        (p.start_iso, p.end_iso, top_n),
    ).fetchall()
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM expenses WHERE expense_date BETWEEN ? AND ?", (p.start_iso, p.end_iso)
    ).fetchone()["n"]

    result.update({
        "total_recorded_expenses": _money(total),
        "expense_entries": count,
        "by_category": [{"category": c, "total": _money(v), "pct_of_expenses": _pct(v, total)}
                        for c, v in by_cat[:top_n]],
        "largest_expenses": [{"date": r["expense_date"], "category": r["category"],
                              "description": r["description"], "amount": _money(r["amount"])} for r in largest],
        "impact_on_profit": {
            "total_sales": fin["total_sales"],
            "gross_profit": fin["gross_profit"],
            "profit_after_recorded_expenses": fin["profit_after_recorded_expenses"],
            "expenses_pct_of_sales": _pct(total, fin["total_sales"]),
            "expenses_pct_of_gross_profit": _pct(total, fin["gross_profit"]) if fin["gross_profit"] > 0 else None,
            "gross_profit_covers_expenses": fin["gross_profit"] >= total,
        },
        "definitions": {k: DEFINITIONS[k] for k in ("gross_profit", "profit_after_recorded_expenses")},
    })
    return result


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #
_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _cur(x: float) -> str:
    return f"{x:,.2f}"


def get_business_recommendations(conn, period=None, start_date=None, end_date=None, today=None) -> dict:
    """Rule-based trends, problems and opportunities computed from the recorded data."""
    today = today or date.today()
    p = resolve_period(period, start_date, end_date, today)
    prev_p = previous_period(p)

    thresholds = {
        "revenue_change_alert_pct": REVENUE_CHANGE_ALERT_PCT,
        "low_gross_margin_pct": LOW_MARGIN_PCT,
        "margin_drop_points": MARGIN_DROP_POINTS,
        "expenses_to_gross_profit_warning": EXPENSE_TO_GROSS_PROFIT_WARN,
        "expense_category_jump_pct": EXPENSE_CATEGORY_JUMP_PCT,
        "dues_to_sales_warning_pct": DUES_TO_SALES_WARN_PCT,
        "overdue_days": OVERDUE_DAYS,
    }
    if _is_database_empty(conn):
        return {"status": "no_data", "period": p.to_dict(), "insights": [],
                "message": "There is no business data yet, so no recommendations can be made."}

    cur = _financials(conn, p.start_iso, p.end_iso)
    prev = _financials(conn, prev_p.start_iso, prev_p.end_iso) if prev_p else None
    insights: list[dict] = []

    def add(code, kind, severity, title, detail, evidence):
        insights.append({"code": code, "type": kind, "severity": severity, "title": title,
                         "detail": detail, "evidence": evidence})

    # 1. Sales activity and trend --------------------------------------------------
    if cur["sale_count"] == 0:
        add("no_sales", "problem", "medium", "No sales recorded in this period",
            "No sales were recorded in the selected period.", {"sale_count": 0})
    elif prev and prev["total_sales"] > 0:
        chg = _pct_change(cur["total_sales"], prev["total_sales"])
        ev = {"current_total_sales": cur["total_sales"], "previous_total_sales": prev["total_sales"],
              "change_pct": chg}
        if chg <= -REVENUE_CHANGE_ALERT_PCT:
            sev = "high" if chg <= -REVENUE_DROP_HIGH_PCT else "medium"
            add("revenue_down", "problem", sev, "Sales fell versus the previous period",
                f"Sales were {_cur(cur['total_sales'])} versus {_cur(prev['total_sales'])} in the previous "
                f"period ({chg:+.1f}%).", ev)
        elif chg >= REVENUE_CHANGE_ALERT_PCT:
            add("revenue_up", "trend", "low", "Sales grew versus the previous period",
                f"Sales were {_cur(cur['total_sales'])} versus {_cur(prev['total_sales'])} in the previous "
                f"period ({chg:+.1f}%).", ev)

    # 2. Margins --------------------------------------------------------------------
    if cur["total_sales"] > 0:
        m = cur["gross_margin_pct"]
        if m is not None and m < LOW_MARGIN_PCT:
            add("low_margin", "problem", "high", "Gross margin is very low",
                f"Gross margin is {m:.1f}% (threshold {LOW_MARGIN_PCT:.0f}%). Check selling prices and costs.",
                {"gross_margin_pct": m})
        if prev and prev["total_sales"] > 0 and prev["gross_margin_pct"] is not None and m is not None:
            drop = prev["gross_margin_pct"] - m
            if drop >= MARGIN_DROP_POINTS:
                add("margin_down", "problem", "medium", "Gross margin declined",
                    f"Gross margin fell from {prev['gross_margin_pct']:.1f}% to {m:.1f}%.",
                    {"current_margin_pct": m, "previous_margin_pct": prev["gross_margin_pct"]})

    # 3. Expenses vs profit -----------------------------------------------------------
    if cur["profit_after_recorded_expenses"] < 0:
        add("loss_after_expenses", "problem", "high", "Recorded expenses exceed gross profit",
            f"Gross profit was {_cur(cur['gross_profit'])} but recorded expenses were "
            f"{_cur(cur['total_recorded_expenses'])}, leaving {_cur(cur['profit_after_recorded_expenses'])}.",
            {"gross_profit": cur["gross_profit"], "expenses": cur["total_recorded_expenses"],
             "profit_after_recorded_expenses": cur["profit_after_recorded_expenses"]})
    elif cur["gross_profit"] > 0 and cur["total_recorded_expenses"] / cur["gross_profit"] > EXPENSE_TO_GROSS_PROFIT_WARN:
        ratio = cur["total_recorded_expenses"] / cur["gross_profit"] * 100
        add("expense_ratio_high", "problem", "medium", "Expenses absorb a large share of gross profit",
            f"Recorded expenses are {ratio:.1f}% of gross profit.",
            {"expenses": cur["total_recorded_expenses"], "gross_profit": cur["gross_profit"],
             "expenses_pct_of_gross_profit": round(ratio, 2)})

    if prev_p:
        cur_cats = _expense_by_category(conn, p.start_iso, p.end_iso)
        prev_cats = _expense_by_category(conn, prev_p.start_iso, prev_p.end_iso)
        jumps = []
        for cat, v in cur_cats.items():
            pv = prev_cats.get(cat, 0.0)
            if pv > 0 and v > pv and (v - pv) / pv * 100 >= EXPENSE_CATEGORY_JUMP_PCT:
                jumps.append((v - pv, cat, v, pv))
        for inc, cat, v, pv in sorted(jumps, reverse=True)[:3]:
            add("expense_category_up", "problem", "medium", f"'{cat}' expenses increased",
                f"{cat} expenses rose from {_cur(pv)} to {_cur(v)} ({(v - pv) / pv * 100:+.1f}%).",
                {"category": cat, "current": _money(v), "previous": _money(pv)})

    # 4. Inventory ---------------------------------------------------------------------
    analysis = _stock_analysis(conn, today)
    urgent = sorted((a for a in analysis if a["priority"] in ("critical", "high")), key=_priority_sort_key)
    if urgent:
        sev = "high" if any(a["priority"] == "critical" for a in urgent) else "medium"
        add("restock_urgent", "problem", sev, "Fast-selling products are running out",
            f"{len(urgent)} product(s) are out of stock or will sell out within about "
            f"{SOON_COVER_DAYS} days at the recent sales rate.",
            {"products": [{"name": a["name"], "stock_qty": a["stock_qty"], "days_of_cover": a["days_of_cover"],
                           "suggested_reorder_qty": a["suggested_reorder_qty"]} for a in urgent[:5]]})
    slow = sorted((a for a in analysis if a["stock_qty"] > 0 and a["units_sold_last_30_days"] == 0),
                  key=lambda a: -(a["stock_qty"] * a["cost_price"]))
    slow_value = sum(a["stock_qty"] * a["cost_price"] for a in slow)
    if slow and slow_value > 0:
        add("slow_movers", "problem", "low", "Cash is tied up in products that are not selling",
            f"{len(slow)} product(s) had no sales in the last {VELOCITY_WINDOW_DAYS} days; stock worth "
            f"{_cur(slow_value)} at cost. Consider a promotion or bundle.",
            {"stock_value_at_cost": _money(slow_value),
             "top_products": [{"name": a["name"], "stock_qty": a["stock_qty"],
                               "stock_value_at_cost": _money(a["stock_qty"] * a["cost_price"])} for a in slow[:3]]})

    # 5. Customer dues -----------------------------------------------------------------
    dues = _compute_dues(conn, today)
    owing = [r for r in dues["records"] if r["_raw_balance"] > 0.005]
    total_due = sum(r["_raw_balance"] for r in owing)
    if total_due > 0 and cur["total_sales"] > 0:
        ratio = total_due / cur["total_sales"] * 100
        if ratio >= DUES_TO_SALES_WARN_PCT:
            sev = "high" if ratio >= DUES_TO_SALES_HIGH_PCT else "medium"
            top = sorted(owing, key=lambda r: -r["_raw_balance"])[:3]
            add("high_dues", "problem", sev, "A large amount is owed by customers",
                f"Customers owe {_cur(total_due)}, equal to {ratio:.1f}% of this period's sales. "
                "Collecting it would improve cash flow.",
                {"total_outstanding": _money(total_due), "pct_of_period_sales": round(ratio, 2),
                 "top_debtors": [{"name": r["name"], "outstanding_balance": r["outstanding_balance"]} for r in top]})
    overdue = sum(r["_over_60"] for r in owing)
    if overdue > 0.005:
        add("overdue_dues", "problem", "medium", f"Customer balances older than {OVERDUE_DAYS} days",
            f"{_cur(overdue)} has been unpaid for more than {OVERDUE_DAYS} days.",
            {"overdue_amount": _money(overdue)})

    # 6. Pricing problems and best performers --------------------------------------------
    below = conn.execute(
        """
        SELECT COUNT(*) AS lines, COALESCE(SUM((si.unit_cost - si.unit_price) * si.quantity), 0.0) AS loss
        FROM sale_items si JOIN sales s ON s.id = si.sale_id
        WHERE s.sale_date BETWEEN ? AND ? AND si.unit_price < si.unit_cost
        """,
        (p.start_iso, p.end_iso),
    ).fetchone()
    if below["lines"] > 0:
        add("sold_below_cost", "problem", "high", "Some items were sold below cost",
            f"{below['lines']} sale line(s) were priced below cost, losing {_cur(below['loss'])} in total.",
            {"sale_lines": below["lines"], "total_loss": _money(below["loss"])})

    top = get_top_products(conn, start_date=p.start_iso, end_date=p.end_iso, limit=1, sort_by="profit", today=today)
    if top["products"] and top["products"][0]["gross_profit"] > 0 and cur["gross_profit"] > 0:
        t = top["products"][0]
        share = t["gross_profit"] / cur["gross_profit"] * 100
        add("top_profit_product", "opportunity", "low", f"'{t['name']}' is the biggest profit driver",
            f"It produced {_cur(t['gross_profit'])} gross profit ({share:.1f}% of the total). "
            "Keep it well stocked and consider promoting it.",
            {"name": t["name"], "gross_profit": t["gross_profit"], "share_of_gross_profit_pct": round(share, 2)})

    insights.sort(key=lambda i: _SEVERITY_RANK[i["severity"]])
    result = {
        "status": "ok" if insights else "no_findings",
        "period": p.to_dict(),
        "compared_with": prev_p.to_dict() if prev_p else None,
        "insight_count": len(insights),
        "insights": insights,
        "thresholds_used": thresholds,
        "notes": ["Insights are rule-based and computed from recorded data only.",
                  "Inventory and dues insights always use current stock and balances; "
                  f"demand is estimated from the last {VELOCITY_WINDOW_DAYS} days."],
    }
    if not insights:
        result["message"] = "No notable problems or trends were detected with the recorded data."
    return result


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
REPORT_TYPES = ("daily", "weekly", "monthly")


def get_business_report(conn, report_type="daily", report_date=None, today=None) -> dict:
    """Daily (that day), weekly (7 days ending on the date) or monthly (month-to-date) report."""
    today = today or date.today()
    if report_type not in REPORT_TYPES:
        raise ToolInputError(f"report_type must be one of: {', '.join(REPORT_TYPES)}.")
    ref = parse_date(report_date, "report_date") if report_date else today
    if ref > today:
        raise ToolInputError("report_date cannot be in the future.")

    if report_type == "daily":
        period = Period("daily", ref, ref)
    elif report_type == "weekly":
        period = Period("weekly", ref - timedelta(days=6), ref)
    else:
        period = Period("monthly", ref.replace(day=1), ref)
    prev = previous_period(period)

    cur = _financials(conn, period.start_iso, period.end_iso)
    prv = _financials(conn, prev.start_iso, prev.end_iso)

    comparison = {}
    for key in ("total_sales", "gross_profit", "total_recorded_expenses",
                "profit_after_recorded_expenses", "net_recorded_cash_flow"):
        comparison[key] = {"current": cur[key], "previous": prv[key],
                           "change_pct": _pct_change(cur[key], prv[key])}

    kwargs = {"start_date": period.start_iso, "end_date": period.end_iso, "today": today}
    top = get_top_products(conn, limit=5, sort_by="profit", **kwargs)
    expenses = get_expense_summary(conn, top_n=5, **kwargs)
    low = get_low_stock_products(conn, today=today)
    dues = get_customer_dues(conn, limit=5, today=today)
    recs = get_business_recommendations(conn, **kwargs)

    report = {
        "status": "ok",
        "report_type": report_type,
        "report_date": ref.isoformat(),
        "period": period.to_dict(),
        "previous_period": prev.to_dict(),
        "summary": cur,
        "comparison_with_previous_period": comparison,
        "daily_breakdown": _daily_breakdown(conn, period.start_iso, period.end_iso) if report_type != "daily" else None,
        "top_products_by_profit": top["products"],
        "expenses": {k: expenses.get(k) for k in ("total_recorded_expenses", "by_category", "largest_expenses",
                                                  "impact_on_profit") if k in expenses},
        "inventory": {
            "low_stock_count": low.get("low_stock_count", 0),
            "out_of_stock_count": low.get("out_of_stock_count", 0),
            "most_urgent": low.get("products", [])[:5],
        },
        "customer_dues": {
            "total_outstanding": dues.get("total_outstanding", 0.0),
            "customers_owing_count": dues.get("customers_owing_count", 0),
            "top_debtors": dues.get("customers", [])[:5],
        },
        "insights": recs.get("insights", []),
        "definitions": DEFINITIONS,
        "notes": ["Inventory and customer dues show the position as of the generating date, not the report date."],
    }
    if cur["sale_count"] == 0 and cur["total_recorded_expenses"] == 0:
        report["status"] = "no_data"
        report["message"] = "No sales or expenses are recorded for this report period."
    return report


def report_to_text(report: dict) -> str:
    """Plain-text rendering of get_business_report output (no numbers are recomputed)."""
    if report.get("status") == "error":
        return f"Error: {report.get('message')}"
    s, per = report["summary"], report["period"]
    lines = [
        f"{report['report_type'].upper()} BUSINESS REPORT  ({per['start_date']} to {per['end_date']})",
        "=" * 60,
    ]
    if report.get("message"):
        lines += [report["message"], ""]
    lines += [
        f"Sales recorded:                  {s['sale_count']}",
        f"Total sales:                     {s['total_sales']:,.2f}",
        f"Cost of goods sold:              {s['cost_of_goods_sold']:,.2f}",
        f"Gross profit:                    {s['gross_profit']:,.2f}",
        f"Recorded expenses:               {s['total_recorded_expenses']:,.2f}",
        f"Profit after recorded expenses:  {s['profit_after_recorded_expenses']:,.2f}",
        f"Cash received:                   {s['total_cash_received']:,.2f}",
        f"Net recorded cash flow:          {s['net_recorded_cash_flow']:,.2f}",
        "",
        "Change vs previous period:",
    ]
    for key, c in report["comparison_with_previous_period"].items():
        pct = "n/a" if c["change_pct"] is None else f"{c['change_pct']:+.1f}%"
        lines.append(f"  {key}: {c['current']:,.2f} (previous {c['previous']:,.2f}, {pct})")
    if report["top_products_by_profit"]:
        lines += ["", "Top products by gross profit:"]
        lines += [f"  {p['name']}: {p['units_sold']} units, profit {p['gross_profit']:,.2f}"
                  for p in report["top_products_by_profit"]]
    inv = report["inventory"]
    lines += ["", f"Low-stock products: {inv['low_stock_count']} (out of stock: {inv['out_of_stock_count']})"]
    lines += [f"  [{i['priority']}] {i['name']}: {i['stock_qty']} left, reorder {i['suggested_reorder_qty']}"
              for i in inv["most_urgent"]]
    dues = report["customer_dues"]
    lines += ["", f"Customer dues outstanding: {dues['total_outstanding']:,.2f} "
                  f"({dues['customers_owing_count']} customers)"]
    if report["insights"]:
        lines += ["", "Insights:"]
        lines += [f"  [{i['severity']}] {i['title']}: {i['detail']}" for i in report["insights"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tool registry (used by the local Ollama agent and tests)
# --------------------------------------------------------------------------- #
def _period_properties() -> dict:
    return {
        "period": {"type": "string", "enum": list(PERIOD_NAMES),
                   "description": f"Named period. Defaults to {DEFAULT_PERIOD}. Ignored if start_date is given."},
        "start_date": {"type": "string", "description": "Custom range start, YYYY-MM-DD (inclusive)."},
        "end_date": {"type": "string", "description": "Custom range end, YYYY-MM-DD (inclusive). Defaults to today."},
    }


TOOLS: dict[str, dict[str, Any]] = {
    "get_sales_summary": {
        "func": get_sales_summary,
        "description": "Total sales, cost of goods sold, gross profit, profit after recorded expenses and recorded "
                       "cash flow for a period, with a daily breakdown for short periods. Use for any question "
                       "about sales, revenue, gross profit, net profit or cash flow. For today's results, use "
                       "period='today'; do not provide only end_date.",
        "parameters": {"type": "object", "properties": _period_properties(), "additionalProperties": False},
    },
    "get_top_products": {
        "func": get_top_products,
        "description": "Best-selling products in a period ranked by revenue, gross profit or units.",
        "parameters": {"type": "object", "properties": {
            **_period_properties(),
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Number of products (default 5)."},
            "sort_by": {"type": "string", "enum": list(SORT_OPTIONS)},
        }, "additionalProperties": False},
    },
    "get_inventory_status": {
        "func": get_inventory_status,
        "description": "Current stock level, stock status and stock value of all products, optionally for one category.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "description": "Optional category name filter."},
            "include_inactive": {"type": "boolean", "description": "Include discontinued products (default false)."},
        }, "additionalProperties": False},
    },
    "get_low_stock_products": {
        "func": get_low_stock_products,
        "description": "Products at or below their reorder level, with priority, days of stock left and suggested "
                       "reorder quantity.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "get_restock_recommendations": {
        "func": get_restock_recommendations,
        "description": "Prioritised restocking list (critical/high/medium/low) with suggested quantities and "
                       "estimated cost. Includes products that will sell out within about a week.",
        "parameters": {"type": "object", "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        }, "additionalProperties": False},
    },
    "get_customer_dues": {
        "func": get_customer_dues,
        "description": "How much customers owe (outstanding credit balances), largest first, with aging. "
                       "Optionally filter by customer name.",
        "parameters": {"type": "object", "properties": {
            "customer_name": {"type": "string", "description": "Full or partial customer name."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        }, "additionalProperties": False},
    },
    "get_expense_summary": {
        "func": get_expense_summary,
        "description": "Expenses for a period by category, the largest individual expenses, and their impact on "
                       "gross profit and profit after recorded expenses.",
        "parameters": {"type": "object", "properties": {
            **_period_properties(),
            "top_n": {"type": "integer", "minimum": 1, "maximum": 50},
        }, "additionalProperties": False},
    },
    "get_business_report": {
        "func": get_business_report,
        "description": "A full daily, weekly (7 days) or monthly (month-to-date) performance report with comparison "
                       "to the previous period, top products, expenses, low stock, dues and insights.",
        "parameters": {"type": "object", "properties": {
            "report_type": {"type": "string", "enum": list(REPORT_TYPES)},
            "report_date": {"type": "string", "description": "Report end date YYYY-MM-DD (default today)."},
        }, "required": ["report_type"], "additionalProperties": False},
    },
    "get_business_recommendations": {
        "func": get_business_recommendations,
        "description": "Data-driven trends, problems and opportunities (sales trend, margins, expenses, restocking, "
                       "slow-moving stock, customer dues, below-cost sales) for a period.",
        "parameters": {"type": "object", "properties": _period_properties(), "additionalProperties": False},
    },
}

TOOLS.update({
    "get_sales_by_date": {
        "func": lambda conn, sale_date, today=None: get_sales_summary(
            conn, start_date=sale_date, end_date=sale_date, today=today,
        ),
        "description": "Sales, COGS, gross profit, recorded expenses and cash flow for one exact date.",
        "parameters": {"type": "object", "properties": {
            "sale_date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
        }, "required": ["sale_date"], "additionalProperties": False},
    },
    "get_top_selling_products": {
        "func": get_top_products,
        "description": "Products ranked by sales revenue, gross profit or units for the requested period.",
        "parameters": TOOLS["get_top_products"]["parameters"],
    },
    "get_inventory": {
        "func": get_inventory_status,
        "description": "Current product stock, low-stock state and inventory value.",
        "parameters": TOOLS["get_inventory_status"]["parameters"],
    },
    "get_low_stock": {
        "func": get_low_stock_products,
        "description": "Active products at or below reorder level, sorted by urgency.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "compare_business_periods": {
        "func": lambda conn, current_period, comparison_period, today=None: {
            "status": "ok",
            "current": get_sales_summary(conn, period=current_period, today=today),
            "comparison": get_sales_summary(conn, period=comparison_period, today=today),
        },
        "description": "Compare financial results for two named periods. Use this month vs last month when asked.",
        "parameters": {"type": "object", "properties": {
            "current_period": {"type": "string", "enum": list(PERIOD_NAMES)},
            "comparison_period": {"type": "string", "enum": list(PERIOD_NAMES)},
        }, "required": ["current_period", "comparison_period"], "additionalProperties": False},
    },
    "generate_business_report": {
        "func": get_business_report,
        "description": TOOLS["get_business_report"]["description"],
        "parameters": TOOLS["get_business_report"]["parameters"],
    },
})


def get_tool_definitions() -> list[dict]:
    """Tool schemas in the format accepted by Ollama's native chat API."""
    return [
        {"type": "function",
         "function": {"name": name, "description": spec["description"], "parameters": spec["parameters"]}}
        for name, spec in TOOLS.items()
    ]


def _error(error_type: str, message: str) -> dict:
    return {"status": "error", "error_type": error_type, "message": message}


def execute_tool(conn: sqlite3.Connection, name: str, arguments: Any, today: date | None = None) -> dict:
    """Validate and run one tool. Never raises: failures come back as {"status": "error", ...}."""
    spec = TOOLS.get(name)
    if spec is None:
        return _error("unknown_tool", f"Unknown tool '{name}'. Available tools: {', '.join(TOOLS)}.")

    if arguments is None or arguments == "":
        args: Any = {}
    elif isinstance(arguments, str):
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            return _error("invalid_input", "Tool arguments were not valid JSON.")
    else:
        args = arguments
    if not isinstance(args, dict):
        return _error("invalid_input", "Tool arguments must be a JSON object.")

    allowed = set(spec["parameters"]["properties"])
    unexpected = sorted(set(args) - allowed)
    if unexpected:
        return _error("invalid_input", f"Unexpected argument(s): {', '.join(unexpected)}. "
                                       f"Allowed: {', '.join(sorted(allowed)) or 'none'}.")
    missing = [k for k in spec["parameters"].get("required", []) if k not in args]
    if missing:
        return _error("invalid_input", f"Missing required argument(s): {', '.join(missing)}.")

    try:
        return spec["func"](conn, **args, today=today)
    except ToolInputError as exc:
        return _error("invalid_input", str(exc))
    except sqlite3.OperationalError as exc:
        logger.exception("Database error in tool %s", name)
        if "no such table" in str(exc).lower():
            return _error("database_error", "The business database has not been initialised (tables are missing).")
        return _error("database_error", "The business database could not be queried.")
    except sqlite3.Error:
        logger.exception("Database error in tool %s", name)
        return _error("database_error", "The business database could not be queried.")
    except Exception:  # noqa: BLE001 - tools must never crash the agent loop
        logger.exception("Unexpected error in tool %s", name)
        return _error("internal_error", "An unexpected error occurred while running this tool.")
