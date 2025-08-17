from typing import Dict
import os
from openai import OpenAI

from .database import get_all_clothes
from .trends import fetch_fashion_trends


def suggest_outfit(description: str, db_path: str = "wardrobe.db", model: str = "gpt-4o-mini") -> str:
    """Generate outfit suggestions using the wardrobe and a language model."""
    clothes = get_all_clothes(db_path)
    trends = fetch_fashion_trends()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    clothing_info = "\n".join(
        [f"{c['id']}: {c['tags']}" for c in clothes]
    )

    prompt = f"""
Kullanıcı açıklaması: {description}
Gardıroptaki öğeler:
{clothing_info}

Güncel moda trendleri:
{trends}

Lütfen uygun kombin önerilerini giysi ID'lerine referans vererek öner.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a personal stylist who suggests outfits from the user's wardrobe."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message["content"].strip()
