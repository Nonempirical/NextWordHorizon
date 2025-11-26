# Next Word Horizon

Probabilistic Horizon Expansion for next-word prediction with visualization capabilities.

## Projektstruktur

```
nextword_horizon/
├── pyproject.toml          # Projektkonfiguration och dependencies
├── README.md               # Denna fil
├── .gitignore              # Git ignore-fil
├── start.bat               # Windows batch startskript
├── start.ps1                # PowerShell startskript
├── start_all.py             # Python startskript (plattformsoberoende)
├── horizon_core/           # Kärnlogik för horizon expansion
│   ├── __init__.py
│   ├── config.py           # Globala inställningar (K, DEPTH, etc.)
│   ├── adapters.py         # ModelAdapter-gränssnitt + implementationer
│   ├── horizon.py          # Probabilistic Horizon Expansion-logik
│   ├── projection.py       # PCA + UMAP-projektion
│   ├── metrics.py          # Entropi, branching factor etc.
│   └── models.py           # Dataklasser för Node, Edge, HorizonResult
├── api/                    # REST API-server
│   ├── __init__.py
│   └── server.py           # FastAPI-server, /expand_horizon endpoint
└── ui/                     # Användargränssnitt
    ├── __init__.py
    └── app.py              # Gradio-app, pratar med API
```

## Installation

### Steg 1: Skapa och aktivera virtuell miljö (rekommenderat)

```bash
# Skapa virtuell miljö
python -m venv venv

# Aktivera virtuell miljö
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Steg 2: Installera projektet

```bash
# Installera projektet med alla dependencies
pip install -e .

# För lokal modellkörning (HuggingFace), installera även torch:
pip install torch
```

**OBS:** Om du använder Python 3.12 eller högre, skapa venv med specifik version:
```bash
# För Python 3.12 (rekommenderat för umap-learn kompatibilitet)
py -3.12 -m venv venv
venv\Scripts\activate
pip install -e .
pip install torch
```

## Användning

### Snabbstart (Rekommenderat)

Använd startskriptet för att starta både API och UI automatiskt:

**Windows:**
```bash
# Dubbelklicka på start.bat eller kör:
start.bat

# Alternativt med PowerShell:
.\start.ps1

# Eller med Python (fungerar på alla plattformar):
python start_all.py
```

Startskriptet kommer att:
- Aktivera virtuell miljö automatiskt
- Starta API-servern i ett separat fönster
- Starta UI:et och öppna det i webbläsaren

### Manuell start

Programmet består av två komponenter som måste köras samtidigt:

#### Terminal 1: Starta API-servern

```bash
# Aktivera virtuell miljö om inte redan aktiv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Starta API-servern
uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

API:et kommer att köras på `http://localhost:8000`

#### Terminal 2: Starta UI:et

```bash
# Aktivera virtuell miljö om inte redan aktiv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Starta Gradio UI
python -m ui.app
```

UI:et kommer att öppnas automatiskt i webbläsaren på `http://localhost:7860`

### Användning i UI:et

1. **Prompt**: Skriv din text (t.ex. "The dog")
2. **Top-K**: Välj antal kandidater (5-50, rekommenderat: 20)
3. **Max Depth**: Välj maximalt djup (1-5, rekommenderat: 3)
4. **Model Backend**: Välj "local" för lokala modeller
5. **Model Name**: Ange modellnamn (t.ex. "gpt2", "Qwen/Qwen2.5-1.5B-Instruct")
6. **Klicka på "Run / Explore"**

**OBS:** Första gången du använder en modell kommer den att laddas ner från HuggingFace (kan ta tid).

**Rekommenderade modeller:**
- `gpt2` - Liten och snabb, bra för testning
- `Qwen/Qwen2.5-1.5B-Instruct` - Modern instruct-modell, bra kvalitet
- `Qwen/Qwen2.5-0.5B-Instruct` - Mindre variant, snabbare

### Testa API:et direkt

Du kan också testa API:et direkt med curl:

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

### Health Check

Testa att API:et är igång:

```bash
curl http://localhost:8000/health
```

### I Google Colab

Installera paketet och kör modulerna direkt i notebook-miljön.

## Krav

- Python 3.11 eller högre (Python 3.12 rekommenderas för bäst kompatibilitet med umap-learn)
- För lokal modellkörning: PyTorch (installeras med `pip install torch`)
- Internetanslutning för att ladda ner modeller från HuggingFace (första gången)
- Minst 4GB RAM för mindre modeller (t.ex. Qwen2.5-1.5B), mer för större modeller

## Felsökning

**Problem:** "ModuleNotFoundError" när du kör programmet
- **Lösning:** Se till att du har installerat projektet med `pip install -e .`

**Problem:** API-servern startar inte
- **Lösning:** Kontrollera att port 8000 inte är upptagen. Ändra port med `--port 8001`

**Problem:** UI:et kan inte ansluta till API:et
- **Lösning:** Kontrollera att API-servern körs och att API URL i UI:et är korrekt (`http://localhost:8000`)

**Problem:** Modell laddas långsamt första gången
- **Lösning:** Detta är normalt. Modeller laddas ner från HuggingFace första gången och cachar lokalt.

**Problem:** "trust_remote_code" varning med Qwen-modeller
- **Lösning:** Detta är normalt och säkert. Qwen-modeller kräver trust_remote_code=True för att fungera korrekt.

**Problem:** Out of memory vid modellkörning
- **Lösning:** Försök använda en mindre modell (t.ex. Qwen2.5-0.5B) eller minska max_depth/top_k parametrarna.

## Ytterligare dokumentation

- `QWEN_GUIDE.md` - Detaljerad guide för att använda Qwen-modeller
- `QUICKSTART_QWEN.md` - Snabbstartsguide för Qwen
- `INSTALLATION_STATUS.md` - Installation status och felsökning

## Utveckling

Projektet är förberett för modern Python (3.11+) och kan köras både lokalt och i Colab.
Python 3.12 rekommenderas för bäst kompatibilitet med alla dependencies.

