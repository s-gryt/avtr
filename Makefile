-include .env
MODEL ?= gemma4:e4b-it-q4_K_M

.PHONY: up down logs pull-model dev

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

pull-model:
	ollama pull $(MODEL)

dev:
	cd app && uv run uvicorn main:app --port 8000 --reload
