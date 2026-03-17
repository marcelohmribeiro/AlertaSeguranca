import os
import sys
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict
from dotenv import load_dotenv

# Carrega variáveis de ambiente (.env)
load_dotenv()

# Adiciona raiz do projeto ao path (quando executado de qualquer lugar)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from google.cloud import firestore

VALID_PLATFORMS = ["twitter", "instagram", "youtube", "reddit"]


def prepare_documents(df: pd.DataFrame, platform: str) -> List[Dict]:
    records = []
    
    for idx, row in df.iterrows():
        doc = {
            "platform": platform,
            "comment": str(row.get("comment", "")).strip(),
            "toxicity": float(row.get("toxicity", 0.0)),
            "ingestedAt": datetime.now(timezone.utc),
        }
        records.append(doc)
    
    return records


def upload_to_firestore(csv_path: str, platform: str, collection_name: str = None) -> Dict:
    # Validações
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
    
    if platform.lower() not in VALID_PLATFORMS:
        raise ValueError(f"Platform inválida. Use: {', '.join(VALID_PLATFORMS)}")
    
    platform = platform.lower()
    collection_name = collection_name or platform
    
    # Lê CSV
    print(f"\n📄 Lendo: {csv_path}")
    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"   Total de comentários: {total}")
    
    if total == 0:
        print("   ⚠️  CSV vazio. Nada a enviar.")
        return {"uploaded": 0, "total": 0}
    
    # Prepara documentos
    print(f"📦 Preparando {total} documentos...")
    records = prepare_documents(df, platform)
    
    # Conecta ao Firestore
    print(f"🔗 Conectando ao Firestore...")
    try:
        client = firestore.Client()
    except Exception as e:
        print(f"❌ Erro ao conectar Firestore: {e}")
        print(f"   Certifique-se que GOOGLE_APPLICATION_CREDENTIALS está configurado")
        raise
    
    # Envia para Firestore com IDs automáticos (.add())
    print(f"📤 Enviando para coleção '{collection_name}'...")
    uploaded = 0
    try:
        for i, doc in enumerate(records):
            # .add() gera ID automaticamente
            client.collection(collection_name).add(doc)
            uploaded += 1
            
            if (i + 1) % 50 == 0:
                print(f"   ✓ {i + 1}/{total}")
        
        print(f"\n✅ Upload concluído:")
        print(f"   ✓ Documentos enviados: {uploaded}")
        print(f"   📍 Coleção: {collection_name}")
        print(f"   🔗 Platform: {platform}")
        
        return {
            "uploaded": uploaded,
            "total": total,
            "collection": collection_name,
            "platform": platform
        }
    except Exception as e:
        print(f"\n❌ Erro ao enviar para Firestore: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python tools/upload_to_firestore.py <arquivo.csv> <platform> [collection_name]")
        print(f"\nPlatforms válidas: {', '.join(VALID_PLATFORMS)}")
        print("\nExemplos:")
        print("  python tools/upload_to_firestore.py data/out/filtered_flagged.csv twitter")
        print("  python tools/upload_to_firestore.py output_flagged.csv instagram my_collection")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    platform = sys.argv[2]
    collection = sys.argv[3] if len(sys.argv) > 3 else None
    
    result = upload_to_firestore(csv_file, platform, collection)
    print(f"\n{result}")
