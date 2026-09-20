# dedupe.py
import os
import json
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sqlalchemy.orm import Session
import model

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GroqAPIKEY"))


def make_lead_text(lead: model.Lead) -> str:
    parts = [
        lead.full_name or "",
        lead.email or "",
        lead.company_name or "",
        lead.phone_number or "",
        lead.country or "",
    ]
    return " ".join(p for p in parts if p).strip()


def get_email_domain(email: str) -> str | None:
    if email and "@" in email:
        return email.split("@")[1].lower().strip()
    return None


def normalize_phone(phone: str) -> str:
    if not phone:
        return ""
    return "".join(c for c in phone if c.isdigit())


def sync_embeddings(leads: list[model.Lead], db: Session) -> dict[int, np.ndarray]:
    existing = {
        e.lead_id: e
        for e in db.query(model.LeadEmbedding).all()
    }

    to_embed = [lead for lead in leads if lead.id not in existing]

    if to_embed:
        texts = [make_lead_text(lead) for lead in to_embed]
        vectors = embed_model.encode(texts, show_progress_bar=True)

        for lead, vector in zip(to_embed, vectors):
            db.add(model.LeadEmbedding(
                lead_id=lead.id,
                embedding=vector.tolist(),
                lead_text=make_lead_text(lead),
            ))
        db.commit()

    all_embeddings = db.query(model.LeadEmbedding).all()
    return {e.lead_id: np.array(e.embedding) for e in all_embeddings}



def build_candidate_pairs(leads: list[model.Lead]) -> list[tuple[int, int]]:
    lead_index = {lead.id: i for i, lead in enumerate(leads)}
    groups = defaultdict(set)

    for lead in leads:
        domain = get_email_domain(lead.email)
        if domain:
            groups[f"domain:{domain}"].add(lead.id)

        phone = normalize_phone(lead.phone_number)
        if len(phone) >= 7:
            groups[f"phone:{phone[-7:]}"].add(lead.id)

    candidate_pairs = set()
    for group_ids in groups.values():
        group_ids = list(group_ids)
        if len(group_ids) < 2:
            continue
        for i in range(len(group_ids)):
            for j in range(i + 1, len(group_ids)):
                pair = tuple(sorted([group_ids[i], group_ids[j]]))
                candidate_pairs.add(pair)

    return list(candidate_pairs)



def generate_reason(lead_a: model.Lead, lead_b: model.Lead) -> str:
    prompt = f"""
You are a data quality assistant. Compare these two leads and explain in ONE concise sentence 
why they might be the same person. Focus on similarities and differences in name, email, phone, and company.

Lead A:
- Name: {lead_a.full_name}
- Email: {lead_a.email}
- Phone: {lead_a.phone_number}
- Company: {lead_a.company_name}
- Country: {lead_a.country}

Lead B:
- Name: {lead_b.full_name}
- Email: {lead_b.email}
- Phone: {lead_b.phone_number}
- Company: {lead_b.company_name}
- Country: {lead_b.country}

Respond with ONE sentence only, no preamble.
""".strip()

    response = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()



def find_dedupe_candidates(db: Session, threshold: float = 0.85) -> list[dict]:
    leads = db.query(model.Lead).all()
    lead_map = {lead.id: lead for lead in leads}

    embedding_map = sync_embeddings(leads, db)
    candidate_pairs = build_candidate_pairs(leads)

    results = []
    for id_a, id_b in candidate_pairs:
        if id_a not in embedding_map or id_b not in embedding_map:
            continue

        vec_a = embedding_map[id_a].reshape(1, -1)
        vec_b = embedding_map[id_b].reshape(1, -1)
        score = float(cosine_similarity(vec_a, vec_b)[0][0])

        if score >= threshold:
            lead_a = lead_map[id_a]
            lead_b = lead_map[id_b]

            reason = generate_reason(lead_a, lead_b)

            results.append({
                "confidence": round(score, 4),
                "reason": reason,
                "leads": [
                    {
                        "id": lead_a.id,
                        "full_name": lead_a.full_name,
                        "email": lead_a.email,
                        "phone_number": lead_a.phone_number,
                        "company_name": lead_a.company_name,
                        "country": lead_a.country,
                    },
                    {
                        "id": lead_b.id,
                        "full_name": lead_b.full_name,
                        "email": lead_b.email,
                        "phone_number": lead_b.phone_number,
                        "company_name": lead_b.company_name,
                        "country": lead_b.country,
                    },
                ],
            })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results