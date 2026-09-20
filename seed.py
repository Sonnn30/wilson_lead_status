import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))

df = pd.read_csv("data_clean.csv")

df = df.rename(columns={
    "Record ID"          : "record_id",
    "First Name"         : "first_name",
    "Last Name"          : "last_name",
    "Full Name"          : "full_name",
    "Job Title"          : "job_title",
    "Company Name"       : "company_name",
    "Email"              : "email",
    "Phone Number"       : "phone_number",
    "Country/Region"     : "country",
    "Lead Status"        : "lead_status",
    "Lifecycle Stage"    : "lifecycle_stage",
    "Original Source"    : "original_source",
    "Contact Owner"      : "contact_owner",
    "Create Date"        : "create_date",
    "Last Modified Date" : "last_modified_date",
    "Notes"              : "notes",
    "Lead Score"         : "lead_score"
})

df["lead_score"] = df["lead_score"].astype(str)  
df["record_id"]  = df["record_id"].astype(int)

df.to_sql(
    name="lead_seed",
    con=engine,
    if_exists="append",
    index=True,           
    index_label="id"      
)

