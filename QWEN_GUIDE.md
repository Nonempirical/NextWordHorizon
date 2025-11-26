# Guide: Använda Qwen/Qwen2.5-1.5B-Instruct

## Översikt

Qwen/Qwen2.5-1.5B-Instruct är nu implementerad och redo att användas med Next Word Horizon-projektet.

## Installation

Modellen kommer automatiskt att laddas ner från HuggingFace första gången du använder den. Se till att du har installerat torch:

```bash
venv\Scripts\activate
pip install torch
```

## Användning

### Via UI

1. Starta API-servern:
```bash
venv\Scripts\activate
uvicorn api.server:app --reload
```

2. Starta UI:et:
```bash
venv\Scripts\activate
python -m ui.app
```

3. I UI:et:
   - **Model Backend**: Välj "local"
   - **Model Name**: Ange `Qwen/Qwen2.5-1.5B-Instruct`
   - Fyll i övriga fält (prompt, top_k, max_depth)
   - Klicka på "Run / Explore"

### Via API direkt

```bash
curl -X POST "http://localhost:8000/expand_horizon" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The dog",
    "top_k": 20,
    "max_depth": 3,
    "model_backend": "local",
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct"
  }'
```

### Via Python-kod

```python
from horizon_core.adapters import LocalHFAdapter
from horizon_core.horizon import expand_horizon
from horizon_core.projection import project_horizon_to_3d

# Skapa adapter
adapter = LocalHFAdapter(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    device="cuda"  # eller "cpu"
)

# Expanderar horizon
result = expand_horizon(
    adapter=adapter,
    prompt="The dog",
    top_k=20,
    max_depth=3
)

# Projektion till 3D
result = project_horizon_to_3d(result)

# Använd resultatet
print(f"Antal noder: {len(result.nodes)}")
print(f"Max depth: {result.max_depth}")
```

## Första gången

Första gången du använder modellen kommer den att laddas ner från HuggingFace (ca 3GB). Detta kan ta några minuter beroende på din internetanslutning.

Modellen cachar lokalt i `~/.cache/huggingface/`, så efter första nedladdningen går det mycket snabbare.

## Tips

- **CUDA**: Om du har en NVIDIA GPU, modellen kommer automatiskt att använda CUDA för snabbare körning
- **CPU**: Modellen fungerar också på CPU, men det kan vara långsammare
- **Minne**: Qwen2.5-1.5B-Instruct kräver cirka 3-4GB RAM/VRAM

## Felsökning

**Problem**: "trust_remote_code" varning
- **Lösning**: Detta är normalt för Qwen-modeller. Koden körs lokalt och är säker.

**Problem**: Modellen laddas långsamt
- **Lösning**: Detta är normalt första gången. Efter nedladdning cachar den lokalt.

**Problem**: Out of memory
- **Lösning**: Försök använda en mindre modell eller minska max_depth/top_k

