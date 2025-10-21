import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import requests

app = FastAPI()

client = OpenAI()  # ключ берётся из переменной окружения OPENAI_API_KEY автоматически

CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")

class Topic(BaseModel):
    topic: str

def get_recent_news(topic: str):
    if not CURRENTS_API_KEY:
        raise HTTPException(status_code=500, detail="Переменная окружения CURRENTS_API_KEY не установлена")

    url = "https://api.currentsapi.services/v1/latest-news"
    params = {
        "language": "en",
        "keywords": topic,
        "apiKey": CURRENTS_API_KEY,
    }
    response = requests.get(url, params=params, timeout=15)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении данных: {response.text}")

    news_data = response.json().get("news", [])
    if not news_data:
        return "Свежих новостей не найдено."
    return "\n".join([article.get("title", "") for article in news_data[:5]])

def chat_once(prompt: str, max_tokens: int, temperature: float = 0.5, stop=None) -> str:
    """Вспомогательная функция для одного запроса к модели"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обращении к OpenAI: {e}")

def generate_content(topic: str):
    recent_news = get_recent_news(topic)

    title = chat_once(
        prompt=(f"Придумайте привлекательный и точный заголовок для статьи на тему '{topic}', "
                f"с учётом актуальных новостей:\n{recent_news}. Заголовок должен быть интересным и ясно "
                f"передавать суть темы."),
        max_tokens=60,
        temperature=0.5,
        stop=["\n"],
    )

    meta_description = chat_once(
        prompt=(f"Напишите мета-описание для статьи с заголовком: '{title}'. "
                f"Оно должно быть полным, информативным и содержать основные ключевые слова."),
        max_tokens=120,
        temperature=0.5,
        stop=["."],
    )

    post_content = chat_once(
        prompt=(
            f"""Напишите подробную статью на тему '{topic}', используя последние новости:
{recent_news}.
Требования:
1) Информативная, логичная
2) Не менее 1500 символов
3) Чёткая структура с подзаголовками
4) Анализ текущих трендов
5) Вступление, основная часть, заключение
6) Примеры из актуальных новостей
7) Каждый абзац не менее 3–4 предложений
8) Легко читаемый и содержательный текст"""
        ),
        max_tokens=1500,
        temperature=0.5,
    )

    return {
        "title": title,
        "meta_description": meta_description,
        "post_content": post_content,
    }

@app.post("/generate-post")
async def generate_post_api(topic: Topic):
    # Проверка ключа OpenAI здесь, чтобы приложение не падало при старте
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Переменная окружения OPENAI_API_KEY не установлена")
    return generate_content(topic.topic)

@app.get("/")
async def root():
    return {"message": "Service is running"}

@app.get("/heartbeat")
async def heartbeat_api():
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

