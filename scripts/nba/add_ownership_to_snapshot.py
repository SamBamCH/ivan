#!/usr/bin/env python3
"""
Add ownership column to ProjectionSnapshot table.

This migration adds the ownership column to store ownership percentages
for locked players (whose games have already started).
"""

from sqlalchemy import text, inspect
from rich.console import Console

from ivan.database.session import init_db, get_session

console = Console()


def main():
    console.print("[bold cyan]Adding ownership column to ProjectionSnapshot table...[/bold cyan]")
    
    engine = init_db('nba')
    
    # Check if column already exists using SQLAlchemy inspector (works for all DB types)
    inspector = inspect(engine)
    
    # Get table name - handle both SQLite (no schema) and PostgreSQL (with schema)
    if engine.dialect.name == 'sqlite':
        table_name = 'projection_snapshot'
        columns = [col['name'] for col in inspector.get_columns(table_name)]
    else:
        # PostgreSQL
        table_name = 'projection_snapshot'
        columns = [col['name'] for col in inspector.get_columns(table_name, schema='nba')]
    
    if 'ownership' in columns:
        console.print("[yellow]⚠ Column 'ownership' already exists in projection_snapshot table[/yellow]")
        return
    
    # Add the column
    console.print("[cyan]• Adding column...[/cyan]")
    
    with engine.connect() as conn:
        if engine.dialect.name == 'sqlite':
            # SQLite doesn't support ALTER TABLE with DEFAULT in the same statement reliably
            conn.execute(text("""
                ALTER TABLE projection_snapshot 
                ADD COLUMN ownership REAL DEFAULT 0.0
            """))
        else:
            # PostgreSQL
            conn.execute(text("""
                ALTER TABLE nba.projection_snapshot 
                ADD COLUMN ownership FLOAT DEFAULT 0.0
            """))
        conn.commit()
    
    console.print("[green]✓ Successfully added ownership column to projection_snapshot table[/green]")


if __name__ == "__main__":
    main()
