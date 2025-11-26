# Snabbstart: Qwen/Qwen2.5-1.5B-Instruct

## ✅ Redo att använda!

Qwen/Qwen2.5-1.5B-Instruct är nu implementerad och redo att användas.

## Snabbstart

### 1. Starta API-servern (Terminal 1)

```bash
venv\Scripts\activate
uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

### 2. Starta UI (Terminal 2)

```bash
venv\Scripts\activate
python -m ui.app
```

### 3. Använd i UI:et

1. Öppna `http://localhost:7860` i webbläsaren
2. Fyll i:
   - **Prompt**: "The dog"
   - **Top-K**: 20
   - **Max Depth**: 3
   - **Model Backend**: `local`
   - **Model Name**: `Qwen/Qwen2.5-1.5B-Instruct`
3. Klicka på **"Run / Explore"**

### Första gången

Första gången kommer modellen att laddas ner från HuggingFace (~3GB). Detta kan ta 5-10 minuter beroende på din internetanslutning.

Du kommer att se meddelanden som:
```
Loading tokenizer for Qwen/Qwen2.5-1.5B-Instruct...
Loading model Qwen/Qwen2.5-1.5B-Instruct...
Model loaded on cpu
```

## Testa direkt i Python

```python
from horizon_core.adapters import LocalHFAdapter
from horizon_core.horizon import expand_horizon
from horizon_core.projection import project_horizon_to_3d

# Skapa adapter (laddar modellen första gången)
adapter = LocalHFAdapter("Qwen/Qwen2.5-1.5B-Instruct")

# Expanderar horizon
result = expand_horizon(
    adapter=adapter,
    prompt="The dog",
    top_k=20,
    max_depth=3
)

# Projektion till 3D
result = project_horizon_to_3d(result)

print(f"Antal noder: {len(result.nodes)}")
print(f"Max depth: {result.max_depth}")
```

## Andra Qwen-modeller

Du kan också använda andra Qwen-modeller:
- `Qwen/Qwen2.5-0.5B-Instruct` (mindre, snabbare)
- `Qwen/Qwen2.5-3B-Instruct` (större, bättre kvalitet)
- `Qwen/Qwen2.5-7B-Instruct` (mycket större, kräver mer minne)

Bara ändra `model_name` i UI:et eller API-anropet.

