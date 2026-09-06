# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python QKD simulation framework. Core package code lives in `qkd/`, with detector, source, protocol, channel, proof, and utility modules split by concern. Proof implementations are under `qkd/proofs/`; reusable helpers are under `qkd/utils/`; lightweight package tests are in `qkd/tests/`. Top-level scripts such as `main.py`, `main_optimized.py`, `run_scm_sweep.py`, and plotting/analysis scripts drive simulations and produce CSV or image outputs. The `scripts/` directory contains verification and example execution scripts, including Chapman figure checks and `scripts/test_run_qkd.py`.

## Build, Test, and Development Commands

- `python3 -m py_compile main_optimized.py qkd/*.py qkd/proofs/*.py`: quick syntax check for the main modules.
- `python3 -m unittest discover qkd/tests`: run package unit tests currently present in `qkd/tests/`.
- `python3 scripts/test_run_qkd.py`: run the broader script-level QKD smoke/test harness.
- `python3 main_optimized.py --workers 1 --min-pulses-log 4 --max-pulses-log 4`: run a small optimized sweep locally.

No project-level `pyproject.toml`, `requirements.txt`, or `Makefile` is currently present, so avoid adding tool assumptions unless you also add the corresponding config.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, type hints where helpful, and dataclasses/enums consistently with existing `qkd.datatypes` patterns. Prefer explicit names such as `pulse_period_ns`, `channel_transmittance`, and `detector_config`; avoid one-letter variables outside tight mathematical expressions. Keep changes focused and preserve module boundaries: detector behavior belongs in `qkd/detectors.py`, protocol flow in `qkd/protocols.py`, and runner orchestration in top-level scripts.

## Testing Guidelines

Add tests near related package tests when practical, using `test_*.py` names. Favor deterministic RNG seeds for simulations and keep smoke tests small enough for local runs. For detector/source changes, include both construction/config checks and at least one small simulation-path check.

## Commit & Pull Request Guidelines

The visible Git history is minimal (`Framework completed`), so use clear imperative commit messages such as `Fix detector enum normalization` or `Add source configuration tests`. Pull requests should summarize the change, list validation commands run, note generated artifacts, and link any issue or experiment being reproduced.

## Agent-Specific Instructions

Do not commit generated CSV, PNG, or cache artifacts unless explicitly requested. Before editing, check for more specific `AGENTS.md` files in subdirectories. Keep large sweeps opt-in; default to small seeded validation runs.
