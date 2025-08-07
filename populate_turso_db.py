import pandas as pd
from redcap import Project
import io
import os
from libsql_client import Client
import asyncio

# This script fetches data from Redcap and populates a Turso database.
# It is intended to be run from the command line.
# The following environment variables must be set:
# - REDCAP_API_K: API key for Redcap project (used for automated QC and labview samples)
# - REDCAP_QC_K: API key for Redcap QC project
# - TURSO_DB_URL: URL of the Turso database (e.g., 'your-db.turso.io')
# - TURSO_AUTH_TOKEN: Auth token for the Turso database

# --- Redcap Data Fetching ---

def get_automated_qc_data():
    """Fetches automated QC data from Redcap."""
    api_url = 'https://redcap.ucsf.edu/api/'
    api_k = os.environ['REDCAP_API_K']
    proj = Project(api_url, api_k)
    f = io.BytesIO(proj.export_file(record='10', field='file')[0])
    df = pd.read_pickle(f)
    return df

def get_qc_status_data():
    """Fetches QC status data from Redcap."""
    api_url = 'https://redcap.ucsf.edu/api/'
    api_k = os.environ['REDCAP_QC_K']
    proj = Project(api_url, api_k)
    df = proj.export_records(format_type='df')
    df = df.reset_index()
    return df

def get_labview_samples_data():
    """Fetches labview samples data from Redcap."""
    api_url = 'https://redcap.ucsf.edu/api/'
    api_k = os.environ['REDCAP_API_K']
    proj = Project(api_url, api_k)
    f = io.BytesIO(proj.export_file(record='8', field='file')[0])
    labview_samples = pd.read_csv(f)
    return labview_samples

# --- Turso Database Logic ---

def get_turso_client():
    """Initializes and returns a Turso database client."""
    url = os.environ.get("TURSO_DB_URL")
    auth_token = os.environ.get("TURSO_AUTH_TOKEN")
    if not url or not auth_token:
        raise ValueError("TURSO_DB_URL and TURSO_AUTH_TOKEN must be set as environment variables.")

    if not url.startswith("libsql://"):
        url = "libsql://" + url.replace("https://", "").replace("http://", "")

    return Client(url, auth_token=auth_token)

async def create_and_populate_tables(client: Client, dataframes: dict):
    """Creates tables and populates them with data."""
    for name, df in dataframes.items():
        print(f"Processing table: {name}")

        await client.execute(f"DROP TABLE IF EXISTS {name}")
        print(f"Dropped existing table (if it existed): {name}")

        df_sanitized = df.astype(str)

        columns = ", ".join([f'"{col}" TEXT' for col in df_sanitized.columns])
        create_table_sql = f"CREATE TABLE {name} ({columns})"
        await client.execute(create_table_sql)
        print(f"Created table: {name}")

        rows = [tuple(row) for row in df_sanitized.where(pd.notna(df_sanitized), None).to_numpy()]

        if not rows:
            print(f"No data to insert for table: {name}")
            continue

        placeholders = ", ".join(["?"] * len(df_sanitized.columns))
        insert_sql = f"INSERT INTO {name} VALUES ({placeholders})"

        batch_args = [(insert_sql, row) for row in rows]

        chunk_size = 100
        for i in range(0, len(batch_args), chunk_size):
            chunk = batch_args[i:i + chunk_size]
            await client.batch(chunk)
            print(f"Inserted chunk {i//chunk_size + 1} into {name}")

        print(f"Inserted {len(rows)} rows into {name}")


async def main():
    """Main function to fetch data and populate the database."""
    print("Fetching data from Redcap...")
    try:
        automated_qc_df = get_automated_qc_data()
        qc_status_df = get_qc_status_data()
        labview_samples_df = get_labview_samples_data()
        print("Data fetched successfully.")
    except Exception as e:
        print(f"Error fetching data from Redcap: {e}")
        return

    dataframes = {
        "automated_qc": automated_qc_df,
        "qc_status": qc_status_df,
        "labview_samples": labview_samples_df,
    }

    print("Connecting to Turso database...")
    try:
        client = get_turso_client()
        print("Connected to Turso database.")
        await create_and_populate_tables(client, dataframes)
        await client.close()
        print("Database population complete.")
    except Exception as e:
        print(f"An error occurred during database operations: {e}")

if __name__ == "__main__":
    asyncio.run(main())
