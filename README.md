# Social Bot

Python 3.12+ modular automation bot for X and Reddit.

## Setup

1. Install dependencies:
   ```bash
   pip install .
   ```
2. Configure `.env` and `config.json` (see `config.example.json`).
3. Authenticate platforms:
   ```bash
   python -m bot login x --account main
   python -m bot login reddit --account main
   ```
4. Run worker:
   ```bash
   python -m bot worker
   ```

## TODOs

- [ ] Integrate locally hosted LLM content generation (see `bot/content/generator.py`).
- [ ] Implement media uploads for X.
