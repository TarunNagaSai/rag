COMPOSE      = docker compose -f infra/docker-compose.yaml
COMPOSE_DEV  = $(COMPOSE) -f infra/docker-compose.dev.yaml

# ── Dev ───────────────────────────────────────────────────────────────────────
dev:
	$(COMPOSE_DEV) up -d --build

dev-recreate:
	$(COMPOSE_DEV) up -d --build --force-recreate

dev-down:
	$(COMPOSE_DEV) down

dev-logs:
	$(COMPOSE_DEV) logs -f

dev-logs-app:
	$(COMPOSE_DEV) logs -f app

dev-logs-frontend:
	$(COMPOSE_DEV) logs -f frontend

# ── Prod ──────────────────────────────────────────────────────────────────────
up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

# ── Utils ─────────────────────────────────────────────────────────────────────
ps:
	$(COMPOSE) ps

restart:
	$(COMPOSE) restart

clean:
	$(COMPOSE_DEV) down -v

.PHONY: dev dev-recreate dev-down dev-logs dev-logs-app dev-logs-frontend up down logs ps restart clean
