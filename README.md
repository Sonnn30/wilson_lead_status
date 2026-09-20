## Tech Stack

- **Language:** Python
- **Framework:** FastAPI
- **Database:** PostgreSQL
- **ORM:** SQLAlchemy
- **Embedding Model:** `all-MiniLM-L6-v2` (via `sentence-transformers`, runs locally)
- **Vector Storage:** pgvector (PostgreSQL extension)
- **LLM Provider:** Groq API (`qwen/qwen3.8-27b`)

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Sonnn30/wilson_lead_status.git
cd wilson_lead_status
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install fastapi uvicorn sqlalchemy psycopg2-binary pgvector sentence-transformers scikit-learn numpy groq python-dotenv
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```dotenv
DATABASE_URL="postgresql://your_user:your_password@localhost:5432/your_db_name"
GroqAPIKEY="your_groq_api_key"
```

> Get a free Groq API key at https://console.groq.com

### 5. Set up PostgreSQL

Make sure PostgreSQL is running, then enable the pgvector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 6. Run database migrations

```bash
python seed.py
```

This will create all tables and seed the database from `leads_seed.csv`.

### 7. Start the server

```bash
uvicorn main:app --reload
```

API will be available at `http://127.0.0.1:8000`
Interactive docs at `http://127.0.0.1:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/leads` | List leads with optional filters |
| GET | `/leads/export` | Export filtered leads as CSV |
| GET | `/leads/{id}` | Get single lead by record_id |
| PATCH | `/leads/{id}` | Update status, owner, or notes |
| POST | `/leads/ingest` | Ingest a new lead or update existing |
| POST | `/leads/dedupe-candidates` | Find duplicate lead candidates |
| POST | `/leads/{id}/extract-source` | Extract lead source from notes |
| GET | `/dashboard` | Lead counts by status and source |

---

## Design Decisions

### Data Modeling

The seed CSV (`leads_seed.csv`) mirrors a raw HubSpot export — messy dates, inconsistent casing in `Lead Status` (`New`, `new`, `NEW`, `" New"`), blank columns, and mixed name formats (`Full Name` vs `First Name` + `Last Name`). Before loading into PostgreSQL, the data was cleaned in `clean_data.ipynb`:

- `Lead Status` normalized to title case and stripped of whitespace
- Dates normalized to ISO format (`YYYY-MM-DD`)
- `Full Name` filled from `First Name` + `Last Name` where missing, and vice versa
- Columns that were entirely blank across all rows (e.g. `Annual Revenue`, `GDPR consent`) were dropped

PostgreSQL was chosen over SQLite for its support of the `pgvector` extension, which is needed for storing lead embeddings.

### Lead Deduplication (POST /leads/dedupe-candidates)

**Approach: Blocking → Embedding + Cosine Similarity → LLM Explanation**

Running a full pairwise comparison across ~2,000 rows would produce ~2,000,000 pairs, which is not tractable — especially if an LLM call is involved. The pipeline is designed in three stages to keep it efficient:

**Stage 1 — Blocking**
Leads are grouped into candidate buckets based on cheap, exact-match signals:
- Same email domain (e.g. both `@acme.com`)
- Same last 7 digits of phone number

Only leads within the same bucket are compared against each other. This reduces the comparison space from O(n²) to a much smaller set.

**Stage 2 — Embedding + Cosine Similarity**
Each lead is represented as a single text string combining name, email, company, phone, and country, then embedded using `all-MiniLM-L6-v2` (a lightweight, locally-run sentence transformer). Embeddings are stored in a `lead_embeddings` table (backed by pgvector) so they don't need to be recomputed on every request. Cosine similarity is computed only for pairs that passed the blocking step. Pairs with similarity ≥ 0.85 are surfaced as duplicate candidates.

**Stage 3 — LLM Explanation**
For each candidate pair that passes the threshold, a short prompt is sent to the Groq API (`qwen/qwen3.8-27b`) asking it to explain in one sentence why the two leads might be the same person. This adds human-readable context to each result without requiring an LLM call for every pair.

**Why this approach:**
- Blocking keeps it tractable at scale
- Embeddings catch semantic similarity that exact string matching misses (e.g. "Acme Corp" vs "Acme Corporation")
- LLM is used sparingly — only for explanation, not for the comparison itself

### Source Extraction (POST /leads/{id}/extract-source)

**Approach: LLM classification**

The `Original Source` column in the seed data is frequently blank or too generic (e.g. `"Offline Sources"`) to be useful. The `Notes` column, however, often contains richer context written in free text (e.g. `"Met him at the SFF booth, scanned our QR code"`).

A rules/regex approach was considered but ruled out — the free-text nature of the notes is too varied and unpredictable for regex to cover reliably. Instead, a single LLM call to Groq (`qwen/qwen3.8-27b`) is used with a structured prompt that constrains the output to one of seven valid channels:

`Website`, `Event`, `LinkedIn`, `Organic Search`, `Referral`, `Manual/Sales`, `Other`

The prompt asks for a JSON response only, and the result is validated after parsing — if the returned channel is not in the valid list, it falls back to `Other`.

---

## Testing

> Results documented after manual testing via Swagger UI (`/docs`)

### GET /leads

| Scenario | Request | Result |
|---|---|---|
| No filter | `GET /leads` | |
| Filter by status | `GET /leads?status=New` | |
| Filter by owner | `GET /leads?owner=...` | |
| Filter by country | `GET /leads?country=Singapore` | |
| Free-text search | `GET /leads?q=acme` | |
| Combined filters | `GET /leads?status=New&country=Singapore` | |

### GET /leads/export

| Scenario | Request | Result |
|---|---|---|
| Export all | `GET /leads/export` | |
| Export filtered | `GET /leads/export?status=New` | |

### GET /leads/{id}

| Scenario | Request | Result |
|---|---|---|
| Valid ID | `GET /leads/1` | |
| ID not found | `GET /leads/99999` | |

### PATCH /leads/{id}

| Scenario | Request Body | Result |
|---|---|---|
| Update status | `{"status": "Qualified"}` | |
| Update owner | `{"owner": "Jane"}` | |
| Update notes | `{"notes": "follow up next week"}` | |
| ID not found | `PATCH /leads/99999` | |

### POST /leads/ingest

| Scenario | Description | Result |
|---|---|---|
| New email | Payload with new email | |
| Existing email | Payload with existing email | |
| Empty payload | `{}` | |

### POST /leads/dedupe-candidates

| Scenario | Description | Result |
|---|---|---|
| Normal run | `POST /leads/dedupe-candidates` | |

### POST /leads/{id}/extract-source

| Scenario | Description | Result |
|---|---|---|
| Lead with notes | Lead that has notes | |
| Lead without notes | Lead with empty notes | |
| ID not found | `POST /leads/99999/extract-source` | |

### GET /dashboard

| Scenario | Request | Result |
|---|---|---|
| Normal | `GET /dashboard` | |

---

## What I'd Do Next

- **Auto-merge duplicates** — currently the dedup endpoint only surfaces candidates; a merge endpoint that consolidates duplicate records would be the natural next step
- **Batch source extraction** — a `POST /leads/extract-source/batch` endpoint to process all leads at once and store the result in an `extracted_channel` column, enabling more accurate dashboard grouping
- **Better dashboard** — use `extracted_channel` instead of `original_source` for the channel breakdown, and add time-series data (leads per week/month)
- **Incremental embedding updates** — currently embeddings are only generated on first call to `/dedupe-candidates`; ideally they'd be updated automatically when a lead is created or modified via `/ingest` or `PATCH`
- **Authentication** — add API key or JWT-based auth before any production use
- **Pagination** — `GET /leads` currently returns all matching records; pagination would be needed at scale

---

## LLM Usage & Cost

- **Provider:** Groq (free tier)
- **Model:** `qwen/qwen3.8-27b`
- **Used for:** Duplicate explanation generation (`/dedupe-candidates`) and source extraction (`/extract-source`)
- **Estimated cost:** $0 — Groq free tier was sufficient for this assignment

# testing scenario
GET /leads
![leads1](./test/leads1.png)
![leads2](./test/leads2.png)
![leads3](./test/leads3.png)

GET /leads by status
![leadsS1](./test/leadstatus1.png)
![leadsS2](./test/leadsstatus2.png)

GET /leads by owner
![leadsO1](./test/leadsowner1.png)
![leadsO2](./test/leadsowner2.png)

GET /leads by country
![leadsC1](./test/leadscountry1.png)
![leadsC2](./test/leadscountry2.png)

GET /leads by q
![leadsQ1](./test/leadsq1.png)
![leadsQ2](./test/leadsq2.png)


GET /leads/export
![leadsQ1](./test/leadexport1.png)
![leadsQ2](./test/leadexport2.png)

GET /leads/export by status
![leadsQ1](./test/leadsexs1.png)
![leadsQ2](./test/leadsexs2.png)

GET /leads/{id}  by id
![leadsQ1](./test/leadsid1.png)
![leadsQ2](./test/leadsid2.png)


PATCH /leads/{id}  
![leadsQ1](./test/patchleads1.png)
![leadsQ2](./test/patchleads2.png)


POST /leads/ingest  (insert)
![leadsQ1](./test/ingest1.png)
![leadsQ2](./test/ingest2.png)

POST /leads/ingest  (update)
![leadsQ1](./test/ingestu1.png)
![leadsQ2](./test/ingestu2.png)


POST /leads/dedupe-candidates
![leadsQ1](./test/dedup1.png)
![leadsQ2](./test/dedup2.png)


POST /leads/{id}/extract-source
![leadsQ1](./test/extract.png)


GET /dashboard
![leadsQ1](./test/dashboard.png)



