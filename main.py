from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query
import model
from database import engine, SessionLocal
from sqlalchemy.orm import Session
from typing import Optional
from sqlalchemy import or_, func
from schema import LeadUpdate, LeadIngest
from fastapi.responses import StreamingResponse
import io
import csv
import json
from dedupe import find_dedupe_candidates, groq_client
from dotenv import load_dotenv
# ini biar bisa baca .env
load_dotenv()

app = FastAPI()

model.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get('/leads')
def get_leads(status: Optional[str] = Query(None, description="status"), owner: Optional[str] = Query(None, description='owner'), country: Optional[str] = Query(None, description='country'), q : Optional[str] = Query(None, description='q'), db: Session = Depends(get_db)):
    query = db.query(model.Lead)
    if status:
        query = query.filter(model.Lead.lead_status == status)

    if owner:
        query = query.filter(model.Lead.contact_owner == owner)

    if country:
        query = query.filter(model.Lead.country == country)

    if q:
        ilike = f"%{q}%"
        query = query.filter(or_(
            model.Lead.full_name.ilike(ilike),
            model.Lead.company_name.ilike(ilike),
            model.Lead.email.ilike(ilike)
        ))

    leads = query.all()
    return leads

@app.get('/leads/export')
def get_leads(status: Optional[str] = Query(None, description="status"), owner: Optional[str] = Query(None, description='owner'), country: Optional[str] = Query(None, description='country'), q : Optional[str] = Query(None, description='q'), db: Session = Depends(get_db)):
    query = db.query(model.Lead)
    if status:
        query = query.filter(model.Lead.lead_status == status)

    if owner:
        query = query.filter(model.Lead.contact_owner == owner)

    if country:
        query = query.filter(model.Lead.country == country)

    if q:
        ilike = f"%{q}%"
        query = query.filter(or_(
            model.Lead.full_name.ilike(ilike),
            model.Lead.company_name.ilike(ilike),
            model.Lead.email.ilike(ilike)
        ))

    leads = query.all()

    output = io.StringIO()
    fieldnames = [
        "id", "record_id", "first_name", "last_name", "full_name",
        "job_title", "company_name", "email", "phone_number", "country",
        "lead_status", "lifecycle_stage", "original_source", "contact_owner",
        "create_date", "last_modified_date", "notes", "lead_score"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for lead in leads:
        writer.writerow({
            "id": lead.id,
            "record_id": lead.record_id,
            "first_name": lead.first_name,
            "last_name": lead.last_name,
            "full_name": lead.full_name,
            "job_title": lead.job_title,
            "company_name": lead.company_name,
            "email": lead.email,
            "phone_number": lead.phone_number,
            "country": lead.country,
            "lead_status": lead.lead_status,
            "lifecycle_stage": lead.lifecycle_stage,
            "original_source": lead.original_source,
            "contact_owner": lead.contact_owner,
            "create_date": lead.create_date,
            "last_modified_date": lead.last_modified_date,
            "notes": lead.notes,
            "lead_score": lead.lead_score,
        })

    output.seek(0)

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=leads_export.csv"}
    )


@app.get('/leads/{id}')
def get_leads_id(id: int, db: Session = Depends(get_db)):
    db = db.query(model.Lead).filter(id == model.Lead.record_id).first()
    if not db:
        raise HTTPException(status_code=404, detail="id tidak ditemukan")
    
    return db


@app.patch('/leads/{id}')
def update_data(id: int, l_update: LeadUpdate, db: Session = Depends(get_db)):
    lead = db.query(model.Lead).filter(model.Lead.record_id == id).first()

    if not lead:
        raise HTTPException(status_code=404, detail='id tidak ditemukan')

    if l_update.owner is not None:
        lead.contact_owner = l_update.owner

    if l_update.status is not None:
        lead.lead_status = l_update.status

    if l_update.notes is not None:
        lead.notes = l_update.notes

    db.commit()
    db.refresh(lead)

    return lead


@app.post('/lead/ingest')
def ingest(data: LeadIngest, db: Session = Depends(get_db)):
    existing_lead = db.query(model.Lead).filter(model.Lead.email == data.email).first()

    if existing_lead:
        existing_lead.original_source = data.form_name
        existing_lead.create_date = data.submitted_at
        existing_lead.full_name = data.name
        existing_lead.phone_number = data.phone
        existing_lead.company_name = data.company
        existing_lead.country = data.country
        existing_lead.notes = data.messages

        db.commit()
        db.refresh(existing_lead)
        return {"message" : "berhasil update data"}

    else:
        new_data = model.Lead(
            original_source = data.form_name,
            create_date = data.submitted_at,
            full_name = data.name,
            email = data.email,
            phone_number = data.phone,
            company_name = data.company,
            country = data.country,
            notes = data.messages
        )
        db.add(new_data)
        db.commit()
        db.refresh(new_data)

        return {"message" : "berhasil input data"}

@app.post('/leads/dedupe-candidates')
def dedupe_candidates(db: Session = Depends(get_db)):
    candidates = find_dedupe_candidates(db)
    return {
        "total_candidates": len(candidates),
        "candidates": candidates
    }

@app.post("/leads/{id}/extract-source") 
def source(id: int, db: Session = Depends(get_db)):
    data = db.query(model.Lead).filter(model.Lead.record_id == id).first()
    if not data:
        raise HTTPException(status_code=404, detail="id tidak ditemukan")

    if not data.notes:
        raise HTTPException(status_code=400, detail="notes kosong")

    valid_channels = ['Website', 'Event', 'LinkedIn', 'Organic Search', 'Referral', 'Manual/Sales', 'Other']

    prompt = f"""
    You are a marketing analyst. Based on the note below, extract the lead source.

    Note: "{data.notes}"

    Classify the channel into ONE of these options only:
    {', '.join(valid_channels)}

    Respond in JSON format only, no preamble, no explanation:
    {{"channel": "...", "detail": "..."}}
    """.strip()

    response = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    
    # Hapus markdown code block kalau LLM menambahkannya
    raw = raw.replace("```json", "").replace("```", "").strip()
    
    result = json.loads(raw)

    # Validasi channel yang dikembalikan LLM
    if result["channel"] not in valid_channels:
        result["channel"] = "Other"

    return result

@app.get('/dashboard')
def dashboard(db: Session = Depends(get_db)):
    by_status = db.query(model.Lead.lead_status, func.count(model.Lead.id)).group_by(model.Lead.lead_status).all()
    by_source = db.query(model.Lead.original_source, func.count(model.Lead.id)).group_by(model.Lead.original_source).all()

    return {
        "leads_by_status": {status: count for status, count in by_status},
        "leads_by_source": {status: count for status, count in by_source}
    }