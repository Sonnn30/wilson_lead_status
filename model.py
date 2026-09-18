from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Enum, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from pgvector.sqlalchemy import Vector

# ini fungsinya agar sql alchemy bisa mengidentifikasi class ini adalah table
Base = declarative_base()


class Lead(Base):
    __tablename__ = "lead_seed"
    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(Integer, unique=True)
    first_name = Column(String)
    last_name = Column(String)
    full_name = Column(String)
    job_title = Column(String)
    company_name = Column(String)
    email = Column(String)
    phone_number = Column(String)
    country = Column(String)
    lead_status = Column(String)
    lifecycle_stage = Column(String)
    original_source = Column(String)
    contact_owner = Column(String)
    create_date = Column(String)
    last_modified_date = Column(String)
    notes = Column(String)
    lead_score = Column(String)

    embedding = relationship("LeadEmbedding", back_populates="lead", uselist=False)


class LeadEmbedding(Base):
    __tablename__ = "lead_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lead_id = Column(Integer, ForeignKey("lead_seed.id"), unique=True, nullable=False)
    embedding = Column(Vector(384))   # dimensi all-MiniLM-L6-v2
    lead_text = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    lead = relationship("Lead", back_populates="embedding")