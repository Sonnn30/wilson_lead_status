from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime
from typing import Optional

class LeadUpdate(BaseModel):
    status: Optional[str] = None
    owner: Optional[str] = None
    notes: Optional[str] = None

class LeadIngest(BaseModel):
    form_name: str # original source
    submitted_at: str
    name: str
    email: str
    phone: str
    company: str
    country: str
    messages: str


class ExtractLead(BaseModel):
    channel: str
    detail: str

