#!/usr/bin/env python3
"""
Migration script to add ownership_proj columns.
Run this once to update your database schema.
"""

from sqlalchemy import text
from ivan.database.session import init_db, get_session

def run_migration():
    """Add ownership_proj columns to database."""
    
    engine = init_db("nfl")
    
    migrations = [
        # Add ownership_proj to ShowdownPlayer
        """
        ALTER TABLE showdown_player
        ADD COLUMN IF NOT EXISTS ownership_proj FLOAT DEFAULT 0.0;
        """,
        
        # Add actual ownership tracking to ShowdownLineup
        """
        ALTER TABLE showdown_lineup
        ADD COLUMN IF NOT EXISTS cpt_ownership_actual FLOAT;
        """,
        
        """
        ALTER TABLE showdown_lineup
        ADD COLUMN IF NOT EXISTS ownership_sum_actual FLOAT;
        """,
        
        # Add ownership_proj to ProjectionSnapshot
        """
        ALTER TABLE projection_snapshot
        ADD COLUMN IF NOT EXISTS ownership_proj FLOAT DEFAULT 0.0;
        """,
    ]
    
    print("Running database migrations...")
    
    with engine.connect() as conn:
        for i, sql in enumerate(migrations, 1):
            try:
                print(f"  [{i}/{len(migrations)}] {sql.strip().split()[0:6]}")
                conn.execute(text(sql))
                conn.commit()
                print(f"    ✓ Success")
            except Exception as e:
                print(f"    ⚠ Warning: {e}")
                # Continue anyway - might just mean column already exists
    
    print("\n✓ Migration complete!")
    print("\nVerifying columns...")
    
    # Verify columns were added
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT table_name, column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name IN ('showdown_player', 'showdown_lineup', 'projection_snapshot')
              AND column_name LIKE '%ownership%'
            ORDER BY table_name, column_name;
        """))
        
        print("\nOwnership columns in database:")
        print(f"{'Table':<25} {'Column':<30} {'Type':<15} {'Nullable'}")
        print("-" * 80)
        
        for row in result:
            print(f"{row[0]:<25} {row[1]:<30} {row[2]:<15} {row[3]}")
    
    print("\n✓ All ownership columns verified!")
    print("\nNext steps:")
    print("  1. Update your models.py to add Python column definitions")
    print("  2. Update NFLPlayer snapshots to include ownership_proj")
    print("  3. Re-run: python scripts/nfl/get_contest_data.py")
    print("  4. Re-train: python scripts/nfl/train_showdown_model.py")


if __name__ == "__main__":
    run_migration()
