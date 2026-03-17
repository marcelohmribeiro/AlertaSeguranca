import os
import pandas as pd
import time
from detoxify import Detoxify

_model = None

def _get_model() -> Detoxify:
    """Carrega modelo Detoxify em cache global."""
    global _model
    if _model is None:
        print("📦 Carregando modelo Detoxify...")
        _model = Detoxify("multilingual")
    return _model
    

def get_toxicity_score(text: str) -> float:
    """Calcula score de toxicidade (0.0-1.0) para um texto."""
    if not text or not text.strip():
        return 0.0
    
    try:
        model = _get_model()
        results = model.predict(text)
        
        score = max(
            results.get("toxicity", 0.0),
            results.get("severe_toxicity", 0.0),
            results.get("threat", 0.0),
            results.get("insult", 0.0)
        )
        return float(score)
    except Exception as e:
        print(f"⚠️  Erro ao processar: {e}")
        return 0.0


def apply_filter(
    df: pd.DataFrame,
    threshold: float = 0.7,
    delay: float = 0.05,
    verbose: bool = False
) -> pd.DataFrame:
    if df.empty:
        return df
    
    df = df.copy()
    total = len(df)
    
    if verbose:
        print(f"⏳ Analisando toxicidade ({total} comentários)...")
    
    scores = []
    for i, text in enumerate(df["comment"].fillna("").astype(str)):
        score = get_toxicity_score(text)
        scores.append(score)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"   ✓ {i + 1}/{total}")
        
        time.sleep(delay)
    
    # Adiciona colunas de score e flag
    df["toxicity"] = scores
    df["flagged"] = df["toxicity"] >= threshold
    
    # Retorna apenas o que foi flagged (ofensivos)
    return df[df["flagged"]].reset_index(drop=True)


def _generate_output_path(input_path: str, suffix: str = "_flagged") -> str:
    """Gera caminho de saída com sufixo."""
    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    name_without_ext = filename.replace(".csv", "")
    output_filename = f"{name_without_ext}{suffix}.csv"
    return os.path.join(directory, output_filename)


def filter_csv_file(
    input_path: str,
    threshold: float = 0.7,
    delay: float = 0.1
) -> dict:
    # Valida entrada
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")
    
    # Lê CSV
    df = pd.read_csv(input_path)
    total = len(df)
    print(f"\n📄 Processando: {input_path}")
    print(f"   Total de comentários: {total}")
    
    # Aplica filtro
    filtered_df = apply_filter(df, threshold=threshold, delay=delay, verbose=True)
    
    # Exibe resultados
    safe = total - len(filtered_df)
    print(f"\n✅ Análise concluída:")
    print(f"   ✓ Seguros: {safe}")
    print(f"   ✗ Ofensivos (threshold={threshold}): {len(filtered_df)}")
    
    # Salva apenas ofensivos
    flagged_path = None
    if len(filtered_df) > 0:
        flagged_path = _generate_output_path(input_path)
        filtered_df.to_csv(flagged_path, index=False)
        print(f"   🚨 Arquivo salvo: {flagged_path}")
    else:
        print(f"   ℹ️  Nenhum comentário ofensivo encontrado")
    
    return {
        "total": total,
        "safe": safe,
        "flagged": len(filtered_df),
        "threshold": threshold,
        "flagged_path": flagged_path
    }

if __name__ == "__main__":
    import sys
    input_csv = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    
    filter_csv_file(input_csv, threshold=threshold)