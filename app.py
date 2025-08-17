"""FastAPI application for the personal outfit assistant."""

import os
import json
import itertools
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from openai import OpenAI

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing")

client = OpenAI(api_key=OPENAI_API_KEY)

# Database: default to SQLite for quick start
engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///wardrobe.db"), future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
Base = declarative_base()


# ----------------------------------------------------------------------------
# Database schema
# ----------------------------------------------------------------------------
item_tag = Table(
    "item_tag",
    Base.metadata,
    Column("item_id", Integer, ForeignKey("items.id")),
    Column("tag_id", Integer, ForeignKey("tags.id")),
)


class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    image_url = Column(Text, nullable=True)
    color = Column(String, nullable=True)
    category = Column(String, nullable=True)
    season = Column(String, nullable=True)
    formality = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    tags = relationship("Tag", secondary=item_tag, back_populates="items")


class Tag(Base):
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)
    items = relationship("Item", secondary=item_tag, back_populates="tags")


Base.metadata.create_all(engine)


# ----------------------------------------------------------------------------
# Pydantic schemas
# ----------------------------------------------------------------------------
class ItemIn(BaseModel):
    title: str
    image_url: Optional[str] = None
    color: Optional[str] = None
    category: Optional[str] = None
    season: Optional[str] = None
    formality: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class RecommendIn(BaseModel):
    query: str = Field(..., description="Serbest metin: yer, zaman, etkinlik, hava vb.")
    n_outfits: int = 3
    must_include_tags: List[str] = Field(default_factory=list)
    must_exclude_tags: List[str] = Field(default_factory=list)
    allow_web_trends: bool = True
    locale: str = "tr-TR"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def upsert_tags(db, names: List[str]) -> List[Tag]:
    tags: List[Tag] = []
    for name in {n.strip().lower() for n in names if n.strip()}:
        t = db.query(Tag).filter_by(name=name).one_or_none()
        if not t:
            t = Tag(name=name)
            db.add(t)
        tags.append(t)
    return tags


def search_inventory(db, include: List[str], exclude: List[str]) -> List[Item]:
    q = db.query(Item).join(Item.tags, isouter=True)
    for inc in include:
        q = q.filter(Item.tags.any(Tag.name == inc.lower()))
    for exc in exclude:
        q = q.filter(~Item.tags.any(Tag.name == exc.lower()))
    return q.all()


def canonicalize_item(i: Item) -> dict:
    return {
        "id": i.id,
        "title": i.title,
        "image_url": i.image_url,
        "color": i.color,
        "category": i.category,
        "season": i.season,
        "formality": i.formality,
        "tags": [t.name for t in i.tags],
        "notes": i.notes,
    }


# ----------------------------------------------------------------------------
# FastAPI
# ----------------------------------------------------------------------------
app = FastAPI(title="Kişisel Kombin Asistanı (MVP)")


@app.post("/items")
def add_item(payload: ItemIn):
    db = SessionLocal()
    try:
        item = Item(
            title=payload.title,
            image_url=payload.image_url,
            color=payload.color or None,
            category=payload.category or None,
            season=payload.season or None,
            formality=payload.formality or None,
            notes=payload.notes,
        )
        item.tags = upsert_tags(db, payload.tags)
        db.add(item)
        db.commit()
        db.refresh(item)
        return {"ok": True, "item": canonicalize_item(item)}
    finally:
        db.close()


@app.get("/items")
def list_items():
    db = SessionLocal()
    try:
        items = db.query(Item).all()
        return {"count": len(items), "items": [canonicalize_item(i) for i in items]}
    finally:
        db.close()


# ----------------------------------------------------------------------------
# LLM tool definitions
# ----------------------------------------------------------------------------
def tool_search_inventory(include_tags: List[str], exclude_tags: List[str]) -> dict:
    db = SessionLocal()
    try:
        res = search_inventory(db, include_tags, exclude_tags)
        return {"results": [canonicalize_item(i) for i in res]}
    finally:
        db.close()


def tool_make_outfits(candidates: List[dict], n: int = 3) -> dict:
    by_cat: dict = {}
    for c in candidates:
        by_cat.setdefault(c.get("category") or "diğer", []).append(c)

    tops = by_cat.get("gömlek", []) + by_cat.get("t-shirt", []) + by_cat.get("kazak", []) + by_cat.get("bluz", [])
    bottoms = by_cat.get("pantolon", []) + by_cat.get("jean", []) + by_cat.get("etek", [])
    shoes = by_cat.get("ayakkabı", []) + by_cat.get("sneaker", []) + by_cat.get("bot", [])
    outers = by_cat.get("ceket", []) + by_cat.get("blazer", []) + by_cat.get("mont", []) + by_cat.get("trenchcoat", [])

    combos: List[List[dict]] = []
    for t, b, s in itertools.product(tops or [], bottoms or [], shoes or []):
        base = [t, b, s]
        if outers:
            for o in outers:
                combos.append(base + [o])
        else:
            combos.append(base)
        if len(combos) >= 50:
            break

    return {"outfits": combos[: max(1, n)]}


# ----------------------------------------------------------------------------
# Responses API setup
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Sen Murat'ın kişisel kombin asistanısın.\n"
    "Amaç: Murat'ın metinden verdiği bağlama göre, SADE ve uygulanabilir kombin önerileri üret.\n"
    "Kurallar:\n"
    "- Gardıroptaki parçalara sadık kal. Uydurma ürün önermeyeceksin.\n"
    "- Kullanıcı izin verdiyse web trendlerini kısaca tarayabilir, ama trendi öneriye çevirirken envanterle eşleştir.\n"
    "- Çıkışta JSON döndür: {\"analysis\": \"...\", \"outfits\": [ { \"items\":[item_ids], \"rationale\":\"...\" } ]}\n"
    "- Nerede gerekliyse \"tool\" çağır. Önce filtre etiketi çıkar, sonra envanter ara, sonra kombin oluştur.\n"
    "- Türkçe yanıtla. Kısa ama net yaz."
)


FUNCTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_inventory",
            "description": "Etiketlere göre envanterde parça ara",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_tags": {"type": "array", "items": {"type": "string"}},
                    "exclude_tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["include_tags", "exclude_tags"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_outfits",
            "description": "Aday parçalardan kategoriler çakışmadan kombin setleri oluştur",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidates": {"type": "array", "items": {"type": "object"}},
                    "n": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["candidates"],
            },
        },
    },
]


def call_llm_with_tools(
    user_query: str,
    allow_web_trends: bool,
    must_inc: List[str],
    must_exc: List[str],
    n_outfits: int,
    locale: str,
):
    tools = FUNCTION_TOOLS.copy()
    if allow_web_trends:
        tools = [{"type": "web_search"}] + tools

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "must_include_tags": must_inc,
                    "must_exclude_tags": must_exc,
                    "n_outfits": n_outfits,
                    "locale": locale,
                }
            ),
        },
    ]

    tool_state: dict = {}
    for _ in range(8):
        resp = client.responses.create(
            model=MODEL_NAME,
            input=messages,
            tools=tools,
            tool_choice="auto",
        )
        out = resp.output

        if getattr(resp, "tool_calls", None):
            for tc in resp.tool_calls:
                if tc.type == "function":
                    if tc.name == "search_inventory":
                        args = json.loads(tc.arguments or "{}")
                        results = tool_search_inventory(
                            args.get("include_tags", []),
                            args.get("exclude_tags", []),
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(results),
                            }
                        )
                        tool_state["last_search"] = results
                    elif tc.name == "make_outfits":
                        args = json.loads(tc.arguments or "{}")
                        results = tool_make_outfits(
                            args.get("candidates", []),
                            args.get("n", n_outfits),
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(results),
                            }
                        )
                        tool_state["last_outfits"] = results
                elif tc.type == "web_search" and allow_web_trends:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"ack": True}),
                        }
                    )
            continue

        text = getattr(resp, "output_text", None) or (
            out[0].content[0].text.value if out else ""
        )
        return text or "{}"

    return json.dumps(tool_state or {})


@app.post("/recommend")
def recommend(payload: RecommendIn):
    result_json = call_llm_with_tools(
        user_query=payload.query,
        allow_web_trends=payload.allow_web_trends,
        must_inc=payload.must_include_tags,
        must_exc=payload.must_exclude_tags,
        n_outfits=payload.n_outfits,
        locale=payload.locale,
    )
    try:
        parsed = json.loads(result_json)
    except Exception:
        parsed = {"analysis": result_json, "outfits": []}
    return parsed

