"""Pydantic request/response models for the SmartBiz AI API."""
from datetime import date
from typing import Any, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

PeriodName = Literal[
    "today", "yesterday", "this_week", "last_7_days", "this_month",
    "last_30_days", "last_month", "this_year", "all_time",
]
ReportType = Literal["daily", "weekly", "monthly"]

DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, examples=["How much did I sell today?"])
    history: List[ChatMessage] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class ToolCallInfo(BaseModel):
    name: str
    arguments: Any = None
    status: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    tools_used: List[ToolCallInfo] = Field(default_factory=list)
    model: str


class ErrorResponse(BaseModel):
    detail: str


class ProductWrite(BaseModel):
    sku: str = Field(min_length=1, max_length=50)
    name: str = Field(min_length=1, max_length=150)
    category: str = Field(default="General", min_length=1, max_length=100)
    cost_price: float = Field(ge=0, le=1_000_000_000)
    selling_price: float = Field(ge=0, le=1_000_000_000)
    stock_qty: int = Field(ge=0, le=1_000_000_000)
    reorder_level: int = Field(default=0, ge=0, le=1_000_000_000)
    is_active: bool = True

    @field_validator("sku", "name", "category")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field must not be blank.")
        return value


class CustomerWrite(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    phone: Optional[str] = Field(default=None, max_length=50)

    @field_validator("name")
    @classmethod
    def customer_name_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Customer name must not be blank.")
        return value


class SaleLineWrite(BaseModel):
    product_id: int = Field(gt=0)
    quantity: int = Field(gt=0, le=1_000_000)
    unit_price: Optional[float] = Field(default=None, ge=0, le=1_000_000_000)


class SaleWrite(BaseModel):
    sale_date: date
    customer_id: Optional[int] = Field(default=None, gt=0)
    amount_paid: Optional[float] = Field(default=None, ge=0, le=1_000_000_000)
    notes: Optional[str] = Field(default=None, max_length=500)
    request_id: Optional[UUID] = None
    items: List[SaleLineWrite] = Field(min_length=1, max_length=100)


class PaymentWrite(BaseModel):
    payment_date: date
    amount: float = Field(gt=0, le=1_000_000_000)
    notes: Optional[str] = Field(default=None, max_length=500)


class ExpenseWrite(BaseModel):
    expense_date: date
    category: str = Field(min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    amount: float = Field(gt=0, le=1_000_000_000)

    @field_validator("category")
    @classmethod
    def category_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Expense category must not be blank.")
        return value
