import sqlite3
import json
from typing import Dict, List


def init_db(db_path: str = "wardrobe.db") -> None:
    """Create the wardrobe database if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS clothes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            tags TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def add_clothing(image_path: str, tags: Dict[str, str], db_path: str = "wardrobe.db") -> None:
    """Insert a clothing item into the database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO clothes(image_path, tags) VALUES (?, ?)",
        (image_path, json.dumps(tags)),
    )
    conn.commit()
    conn.close()


def get_all_clothes(db_path: str = "wardrobe.db") -> List[Dict]:
    """Return all clothing items with their metadata."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, image_path, tags FROM clothes")
    rows = cur.fetchall()
    conn.close()
    return [
        {"id": row[0], "image_path": row[1], "tags": json.loads(row[2])}
        for row in rows
    ]
