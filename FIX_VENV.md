# Fixa venv med Python 3.12

Om du får fel med numpy eller umap-learn, kan det bero på att venv använder fel Python-version.

## Återskapa venv med Python 3.12

### Steg 1: Stäng alla processer

Stäng alla terminalfönster och processer som använder venv (API-servern, UI:et, etc.)

### Steg 2: Ta bort gamla venv

```bash
# I PowerShell eller CMD
Remove-Item -Recurse -Force venv
```

Om det inte fungerar (pga låsta filer), stäng alla Python-processer först:
- Stäng alla terminalfönster
- Öppna Task Manager (Ctrl+Shift+Esc)
- Avsluta alla python.exe processer
- Försök igen

### Steg 3: Skapa ny venv med Python 3.12

```bash
py -3.12 -m venv venv
```

### Steg 4: Aktivera och installera

```bash
venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -e .
pip install torch
```

### Steg 5: Verifiera

```bash
venv\Scripts\python.exe --version
# Bör visa: Python 3.12.x

venv\Scripts\python.exe -c "import horizon_core; print('OK')"
# Bör visa: OK
```

## Testa startskriptet

Efter att venv är återskapad:

```bash
.\start.bat
```

eller

```bash
python start_all.py
```

