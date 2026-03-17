import os
import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Any
from src.services.filter_csv import apply_filter

CSV_COLUMNS = ["platform", "comment"]

def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    return text[:80]

def ensure_minimal_columns(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    df = df.copy()
    df["platform"] = platform

    if "comment" not in df.columns:
        if "text" in df.columns:
            df["comment"] = df["text"]
        else:
            df["comment"] = None
    return df[CSV_COLUMNS]


def save_csv(
    df: pd.DataFrame,
    platform: str,
    identifier: str,
    kind: str = None,
    base_dir="data/out",
    filter_toxicity: bool = True,
    toxicity_threshold: float = 0.7
):
    if df.empty:
        return None
    
    # Normaliza colunas
    df_norm = ensure_minimal_columns(df, platform)
    
    # Aplica filtro de toxicidade (middleware)
    if filter_toxicity and len(df_norm) > 0:
        df_norm = apply_filter(
            df_norm,
            threshold=toxicity_threshold,
            delay=0.05,
            verbose=False
        )
        
        # Se nenhum comentário passou no filtro, não salva
        if df_norm.empty:
            return None
    
    # Gera caminho e salva
    safe_id = sanitize_filename(identifier)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if kind:
        folder = os.path.join(base_dir, platform, kind)
    else:
        folder = os.path.join(base_dir, platform)

    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, f"{safe_id}_{timestamp}.csv")
    df_norm.to_csv(file_path, index=False, encoding="utf-8")

    return file_path

def export_comments_batch(
    items: List[Dict[str, Any]],
    platform: str,
    kind: str | None,
    save: bool = False,
    base_dir: str = "data/out",
    filter_toxicity: bool = True,
    toxicity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for item in items:
        item_id = item.get("id")
        comments = item.get("data") or []

        csv_path = None
        if save and comments:
            df = pd.DataFrame.from_records(comments)
            csv_path = save_csv(
                df=df,
                platform=platform,
                identifier=item_id,
                kind=kind,
                base_dir=base_dir,
                filter_toxicity=filter_toxicity,
                toxicity_threshold=toxicity_threshold
            )

        saved_comments = len([c for c in comments if csv_path])
        results.append({
            "id": item_id,
            "comments_count": len(comments),
            "saved_comments": saved_comments if csv_path else 0,
            "csv": csv_path,
        })

    return results