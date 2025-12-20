"""SQLite storage for recipes."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

from .models import Recipe


class RecipeStore:
    """SQLite-backed recipe storage."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _migrate(self) -> None:
        with self._conn() as conn:
            # Check if recipes table exists and has the expected schema
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='recipes'")
            table_exists = cursor.fetchone() is not None

            if table_exists:
                # Check column names
                cursor = conn.execute("PRAGMA table_info(recipes)")
                columns = {row[1] for row in cursor.fetchall()}

                # Legacy schema uses 'json', new schema uses 'data'
                if 'json' in columns and 'data' not in columns:
                    # Use legacy schema - set flag
                    self._use_json_column = True
                else:
                    self._use_json_column = 'data' not in columns
            else:
                # Create new table with 'data' column
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS recipes (
                        id TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self._use_json_column = False

    def list(self) -> List[Recipe]:
        col = "json" if getattr(self, '_use_json_column', False) else "data"
        with self._conn() as conn:
            rows = conn.execute(f"SELECT {col} FROM recipes ORDER BY id").fetchall()
        recipes = []
        for row in rows:
            try:
                recipes.append(Recipe.model_validate_json(row[col]))
            except Exception:
                # Skip invalid recipes (e.g., unsupported backends)
                continue
        return recipes

    def get(self, recipe_id: str) -> Optional[Recipe]:
        col = "json" if getattr(self, '_use_json_column', False) else "data"
        with self._conn() as conn:
            row = conn.execute(f"SELECT {col} FROM recipes WHERE id = ?", (recipe_id,)).fetchone()
        if not row:
            return None
        return Recipe.model_validate_json(row[col])

    def save(self, recipe: Recipe) -> None:
        data = recipe.model_dump_json()
        col = "json" if getattr(self, '_use_json_column', False) else "data"
        with self._conn() as conn:
            if self._use_json_column:
                # Legacy schema
                conn.execute(
                    f"""
                    INSERT INTO recipes (id, {col}, created_at, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(id) DO UPDATE SET {col} = excluded.{col}, updated_at = CURRENT_TIMESTAMP
                    """,
                    (recipe.id, data),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO recipes (id, data, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(id) DO UPDATE SET data = excluded.data, updated_at = CURRENT_TIMESTAMP
                    """,
                    (recipe.id, data),
                )

    def delete(self, recipe_id: str) -> bool:
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM recipes WHERE id = ?", (recipe_id,))
        return cursor.rowcount > 0

    def import_from_json(self, json_path: Path) -> int:
        """Import recipes from a JSON file."""
        data = json.loads(json_path.read_text())
        recipes = data if isinstance(data, list) else [data]
        count = 0
        for r in recipes:
            try:
                recipe = Recipe.model_validate(r)
                self.save(recipe)
                count += 1
            except Exception:
                continue
        return count
