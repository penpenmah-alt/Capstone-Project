import requests
import pandas as pd
import duckdb
from functools import reduce

def fetch_and_store_data():
    print("Fetching World Bank data for USA...")
    indicators = {
        'Population': 'SP.POP.TOTL',
        'GDP_USD': 'NY.GDP.MKTP.CD',
        'Debt_Pct_GDP': 'GC.DOD.TOTL.GD.ZS',
        'Human_Capital_Index': 'HD.HCI.OVRL'
    }

    dfs = []
    for name, code in indicators.items():
        url = f"https://api.worldbank.org/v2/country/USA/indicator/{code}?format=json&per_page=100"
        res = requests.get(url).json()

        if len(res) > 1 and res[1] is not None:
            records = [
                {"Year": int(item["date"]), name: item["value"]}
                for item in res[1] if item["value"] is not None
            ]
            if records:
                dfs.append(pd.DataFrame(records))

    if dfs:
        df_usa = reduce(lambda left, right: pd.merge(left, right, on='Year', how='outer'), dfs)
        df_usa = df_usa.sort_values('Year').reset_index(drop=True)
        df_usa = df_usa.ffill().bfill()

        # Connect to DuckDB database file
        conn = duckdb.connect("data.duckdb")

        # Write DataFrame into DuckDB table
        conn.execute("CREATE TABLE IF NOT EXISTS usa_macro_data AS SELECT * FROM df_usa")
        conn.execute("CREATE OR REPLACE TABLE usa_macro_data AS SELECT * FROM df_usa")

        conn.close()
        print("SUCCESS: Data successfully saved to DuckDB (data.duckdb)!")
    else:
        print("ERROR: Ingestion failed: No data retrieved.")

if __name__ == "__main__":
    fetch_and_store_data()
