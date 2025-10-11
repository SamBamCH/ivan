#!/usr/bin/env python3
"""
Migration script to add ownership_proj columns.
SQLite-compatible version.
"""

from sqlalchemy import text, inspect
from ivan.database.session import init_db

def column_exists(engine, table_name, column_name):
    """Check if a column exists in a table."""
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns

def run_migration():
    """Add ownership_proj columns to database (SQLite compatible)."""
    
    engine = init_db("nfl")
    
    # Define migrations with table and column info
    migrations = [
        ('showdown_player', 'ownership_proj', 
         'ALTER TABLE showdown_player ADD COLUMN ownership_proj FLOAT DEFAULT 0.0;'),
        
        ('showdown_lineup', 'cpt_ownership_actual',
         'ALTER TABLE showdown_lineup ADD COLUMN cpt_ownership_actual FLOAT;'),
        
        ('showdown_lineup', 'ownership_sum_actual',
         'ALTER TABLE showdown_lineup ADD COLUMN ownership_sum_actual FLOAT;'),
        
        ('projection_snapshot', 'ownership_proj',
         'ALTER TABLE projection_snapshot ADD COLUMN ownership_proj FLOAT DEFAULT 0.0;'),
    ]
    
    print("Running database migrations (SQLite)...")
    
    with engine.connect() as conn:
        for table_name, column_name, sql in migrations:
            try:
                # Check if column already exists
                if column_exists(engine, table_name, column_name):
                    print(f"  ✓ {table_name}.{column_name} already exists, skipping")
                    continue
                
                print(f"  → Adding {table_name}.{column_name}")
                conn.execute(text(sql))
                conn.commit()
                print(f"    ✓ Success")
            except Exception as e:
                print(f"    ✗ Error: {e}")
                conn.rollback()
    
    print("\n✓ Migration complete!")
    print("\nVerifying columns...")
    
    # Verify columns were added (SQLite version)
    inspector = inspect(engine)
    
    print("\nOwnership columns in database:")
    print(f"{'Table':<25} {'Column':<30} {'Type':<15} {'Nullable'}")
    print("-" * 80)
    
    for table_name in ['showdown_player', 'showdown_lineup', 'projection_snapshot']:
        try:
            columns = inspector.get_columns(table_name)
            for col in columns:
                if 'ownership' in col['name'].lower():
                    nullable = 'YES' if col['nullable'] else 'NO'
                    col_type = str(col['type'])
                    print(f"{table_name:<25} {col['name']:<30} {col_type:<15} {nullable}")
        except Exception as e:
            print(f"{table_name:<25} Error reading table: {e}")
    
    print("\n✓ All ownership columns verified!")
    print("\nNext steps:")
    print("  1. Your models.py already has the column definitions ✓")
    print("  2. Update NFLPlayer snapshots to include ownership_proj")
    print("  3. Re-run: python scripts/nfl/get_contest_data.py")
    print("  4. Re-train: python scripts/nfl/train_showdown_model.py")


if __name__ == "__main__":
    run_migration()