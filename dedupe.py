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

# Load sekali saja saat startup
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GroqAPIKEY"))


# ── Helper ──────────────────────────────────────────────────────────────────

def make_lead_text(lead: model.Lead) -> str:
    """Gabungkan field penting jadi satu string untuk di-embed."""
    parts = [
        lead.full_name or "",
        lead.email or "",
        lead.company_name or "",
        lead.phone_number or "",
        lead.country or "",
    ]
    return " ".join(p for p in parts if p).strip()


def get_email_domain(email: str) -> str | None:
    """Ambil domain dari email, e.g. john@acme.com → acme.com"""
    if email and "@" in email:
        return email.split("@")[1].lower().strip()
    return None


def normalize_phone(phone: str) -> str:
    """Hapus semua karakter non-digit dari nomor telepon."""
    if not phone:
        return ""
    return "".join(c for c in phone if c.isdigit())


# ── Embedding ────────────────────────────────────────────────────────────────

def sync_embeddings(leads: list[model.Lead], db: Session) -> dict[int, np.ndarray]:
    """
    Pastikan semua lead punya embedding di DB.
    Return dict: {lead.id: vector}
    """
    existing = {
        e.lead_id: e
        for e in db.query(model.LeadEmbedding).all()
    }

    # Lead yang belum punya embedding
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

    # Ambil semua embedding dari DB
    all_embeddings = db.query(model.LeadEmbedding).all()
    return {e.lead_id: np.array(e.embedding) for e in all_embeddings}


# ── Blocking ─────────────────────────────────────────────────────────────────

def build_candidate_pairs(leads: list[model.Lead]) -> list[tuple[int, int]]:
    """
    Blocking: kelompokkan lead berdasarkan email domain atau 7 digit terakhir phone.
    Hanya compare lead dalam grup yang sama → jauh lebih efisien dari O(n²).
    """
    lead_index = {lead.id: i for i, lead in enumerate(leads)}
    groups = defaultdict(set)

    for lead in leads:
        # Block by email domain
        domain = get_email_domain(lead.email)
        if domain:
            groups[f"domain:{domain}"].add(lead.id)

        # Block by last 7 digit phone
        phone = normalize_phone(lead.phone_number)
        if len(phone) >= 7:
            groups[f"phone:{phone[-7:]}"].add(lead.id)

    # Generate pasangan unik dari tiap grup
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


# ── LLM Reason ───────────────────────────────────────────────────────────────

def generate_reason(lead_a: model.Lead, lead_b: model.Lead) -> str:
    """Panggil Groq LLM untuk generate penjelasan kenapa dua lead ini kemungkinan duplikat."""
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


# ── Main dedupe function ──────────────────────────────────────────────────────

def find_dedupe_candidates(db: Session, threshold: float = 0.85) -> list[dict]:
    """
    Pipeline lengkap:
    1. Sync embeddings ke DB
    2. Blocking untuk narrow kandidat
    3. Cosine similarity untuk scoring
    4. LLM untuk generate reason
    """
    leads = db.query(model.Lead).all()
    lead_map = {lead.id: lead for lead in leads}

    # Step 1: Sync embeddings
    embedding_map = sync_embeddings(leads, db)

    # Step 2: Blocking → kandidat pairs
    candidate_pairs = build_candidate_pairs(leads)

    # Step 3: Hitung cosine similarity hanya untuk kandidat pairs
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

            # Step 4: LLM generate reason
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

    # Urutkan dari confidence tertinggi
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results