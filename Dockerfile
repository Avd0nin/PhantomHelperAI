# Используем официальный образ Python
FROM python:3.13-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Ставим Unicode-шрифт для PDF с кириллицей
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код и HTML-файлы
COPY backend/app.py .
COPY backend/ai_core.py .
COPY backend/contest_routes.py .
COPY backend/env_loader.py .
COPY backend/templates/ ./templates/
COPY backend/static ./static
# Указываем порт, который будет использоваться Flask
EXPOSE 5000

# Запускаем приложение
CMD ["python", "app.py"]
