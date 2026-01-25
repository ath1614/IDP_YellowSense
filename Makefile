.PHONY: up down build logs test lint clean

up:
	docker-compose up -d

down:
	docker-compose down

build:
	docker-compose build

logs:
	docker-compose logs -f

test:
	python -m pytest tests/ -v

lint:
	python -m flake8 . --max-line-length=100 --exclude=.git,__pycache__

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete

health:
	@curl -s http://localhost:8000/health | python3 -m json.tool
	@curl -s http://localhost:8001/health | python3 -m json.tool

deploy-ocr:
	scp -r ocr/ infra/ocr_vm_setup.sh tech@ocr-vm:~/
	ssh tech@ocr-vm "cd ~/ocr && pip install -r requirements.txt && sudo systemctl restart ocr-service"

deploy-llm:
	scp -r llm/ infra/llm_vm_setup.sh tech@llm-vm:~/
	ssh tech@llm-vm "cd ~/llm && pip install -r requirements.txt && sudo systemctl restart llm-service"

