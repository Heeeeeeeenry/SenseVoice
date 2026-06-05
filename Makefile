# ============================================================
# SenseVoice STT Makefile
# ============================================================
SERVICE_NAME := stt
IMAGE_NAME   := sensevoice:latest

.PHONY: help build run stop clean logs shell dev

help: ## 显示帮助信息
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## 构建 Docker 镜像
	docker build -t $(IMAGE_NAME) .

build-cpu: ## 构建 CPU 模式镜像（无 GPU）
	docker build -t $(IMAGE_NAME)-cpu \
		--build-arg BASE_IMAGE=python:3.10-slim \
		-f Dockerfile.cpu .

dev: ## 本地开发运行
	python webui_streaming.py

run: ## 启动容器（依赖顶层 docker-compose）
	cd .. && docker compose up -d $(SERVICE_NAME)

stop: ## 停止容器
	cd .. && docker compose stop $(SERVICE_NAME)

restart: ## 重启容器
	cd .. && docker compose restart $(SERVICE_NAME)

logs: ## 查看容器日志
	cd .. && docker compose logs -f $(SERVICE_NAME)

shell: ## 进入容器 shell
	cd .. && docker compose exec $(SERVICE_NAME) bash

clean: ## 清理容器和镜像
	cd .. && docker compose down $(SERVICE_NAME)
	docker rmi $(IMAGE_NAME) 2>/dev/null || true
