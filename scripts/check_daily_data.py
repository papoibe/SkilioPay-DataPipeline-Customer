import os
import sys
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.config import ConfigManager

def check_daily_data():
    """
    Check if data for today exists in Raw layer, Processed layer, and Warehouse.
    """
    print(f"=== CHECKING DATA FOR {datetime.now().strftime('%Y-%m-%d')} ===\n")
    
    # 1. Check Raw Data
    raw_file = f"data/raw/churn_data_{datetime.now().strftime('%Y%m%d')}.parquet"
    if os.path.exists(raw_file):
        print(f"[OK] Raw Data found: {raw_file}")
        try:
            df = pd.read_parquet(raw_file)
            print(f"   Rows: {len(df)}")
        except Exception as e:
            print(f"   [ERROR] Reading file: {e}")
    else:
        print(f"[MISSING] Raw Data not found: {raw_file}")
        
    # 2. Check Processed Data
    proc_file = f"data/processed/churn_processed_{datetime.now().strftime('%Y%m%d')}.parquet"
    if os.path.exists(proc_file):
        print(f"\n[OK] Processed Data found: {proc_file}")
        try:
            df = pd.read_parquet(proc_file)
            print(f"   Rows: {len(df)}")
            print(f"   Cols: {len(df.columns)}")
        except Exception as e:
            print(f"   [ERROR] Reading file: {e}")
    else:
        print(f"\n[MISSING] Processed Data not found: {proc_file}")

    # 3. Check Warehouse
    print(f"\n[INFO] Checking Warehouse (Postgres)...")
    try:
        config = ConfigManager().load_config("config/config.yaml")
        db_config = config['storage']['database']['postgresql']
        db_url = f"postgresql+psycopg2://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        engine = create_engine(db_url)
        with engine.connect() as conn:
            # Check users count
            result = conn.execute("SELECT COUNT(*) FROM churn_prediction.users_processed")
            count = result.fetchone()[0]
            print(f"[OK] Users Validation: {count} rows in 'users_processed'")
            
            # Check features count
            result = conn.execute("SELECT COUNT(*) FROM churn_prediction.features")
            count = result.fetchone()[0]
            print(f"[OK] Features Validation: {count} rows in 'features'")
            
    except Exception as e:
        print(f"[ERROR] Warehouse check failed: {e}")
        
    print("\n=== CHECK COMPLETE ===")

if __name__ == "__main__":
    check_daily_data()
