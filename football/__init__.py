"""ShadyNasty fantasy football pipeline (Fantrax dynasty).

Fork of the fantasy-baseball-pipeline medallion architecture. The baseball
pipeline at the repo root is untouched; this package mirrors its layout under
``football/`` so the two share the repo, the venv, and the single ``.env`` while
staying import-isolated (baseball imports ``silver.*``; football imports
``football.silver.*``).
"""
