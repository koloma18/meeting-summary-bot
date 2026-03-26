import os
import json
import math
import tempfile
import requests
from flask import Flask, request, jsonify
from pydub import AudioSegment
from openai import OpenAI
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

app = Flask(__name__)

# ==================== НАСТРОЙКИ ====================

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
GOOGLE_DOC_ID = os.environ["GOOGLE_DOC_ID"]
GOOGLE_CREDENTIALS_JSON = os.environ["GOOGLE_CREDENTIALS_JSON"]

MAX_CHUNK_SIZE_MB = 24

SUMMARY_PROMPT = """Ты — ассистент, который создаёт резюме рабочих встреч для компании Coin Post (криптомедиа).

На входе ты получаешь транскрипт встречи. На выходе — структурированное резюме в строго определённом формате.

ФОРМАТ РЕЗЮМЕ:

Резюме [тип встречи] с [имена участников]

Продолжительность: [X часов Y минут]

Обсудили:
1. [Первая тема/решение]
   - [Детали, подпункты, ответственные, дедлайны]
   - [Ещё детали]
2. [Вторая тема/решение]
   - [Детали]
3. ...

ПРАВИЛА:
- Определи имена участников из транскрипта. Если имя не удаётся определить, напиши "участник".
- Тип встречи определи из контекста: "планёрки", "собрания", "брейншторма", "созвона".
- Пиши конкретно: кто, что, когда. Не пиши абстрактно.
- Если упоминаются дедлайны, даты, ответственные — обязательно включи их.
- Если упоминаются конкретные инструменты, проекты, клиенты — включи названия.
- Пиши на русском языке.
- Не добавляй ничего, чего не было в транскрипте.
- Нумерация основных пунктов: 1. 2. 3. Подпункты через дефис.
- Не добавляй вступление или заключение, только структурированное резюме.

ТРАНСКРИПТ ВСТРЕЧИ:
{transcript}

ПРОДОЛЖИТЕЛЬНОСТЬ ВСТРЕЧИ: {duration}
"""

openai_client = OpenAI(api_key=OPENAI_API_KEY)
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


# ==================== TELEGRAM HELPERS ====================

def send_message(chat_id, text):
    """Отправить текстовое сообщение в Telegram"""
    requests.post(f"{TELEGRAM_API}/sendMessage", json={
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    })


def download_telegram_file(file_id):
    """Скачать файл из Telegram во временный файл"""
    # Получаем путь к файлу
    resp = requests.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id})
    file_path = resp.json()["result"]["file_path"]

    # Скачиваем
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    response = requests.get(file_url, stream=True)

    # Определяем расширение
    ext = os.path.splitext(file_path)[1] or ".ogg"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    for chunk in response.iter_content(chunk_size=8192):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


# ==================== НАРЕЗКА АУДИО ====================

def split_audio(file_path):
    """Нарезаем аудио на куски по ~24 МБ для Whisper API"""
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000

    # Конвертируем в mp3 для оценки размера
    tmp_full = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    audio.export(tmp_full.name, format="mp3", bitrate="64k")
    file_size_mb = os.path.getsize(tmp_full.name) / (1024 * 1024)
    os.unlink(tmp_full.name)

    if file_size_mb <= MAX_CHUNK_SIZE_MB:
        tmp_single = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        audio.export(tmp_single.name, format="mp3", bitrate="64k")
        return [tmp_single.name], duration_seconds

    num_chunks = math.ceil(file_size_mb / MAX_CHUNK_SIZE_MB)
    chunk_duration_ms = len(audio) // num_chunks

    chunk_files = []
    for i in range(num_chunks):
        start = i * chunk_duration_ms
        end = min((i + 1) * chunk_duration_ms, len(audio))
        chunk = audio[start:end]
        tmp_chunk = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        chunk.export(tmp_chunk.name, format="mp3", bitrate="64k")
        chunk_files.append(tmp_chunk.name)

    return chunk_files, duration_seconds


# ==================== ТРАНСКРИПЦИЯ ====================

def transcribe_audio(file_paths):
    """Транскрибируем каждый кусок через Whisper API"""
    full_transcript = []

    for i, path in enumerate(file_paths):
        print(f"Транскрибируем кусок {i + 1}/{len(file_paths)}...")
        with open(path, "rb") as audio_file:
            response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ru",
                prompt=(
                    "Coin Post, криптовалюта, блокчейн, DeFi, GameFi, "
                    "токенсейл, Telegram, YouTube, AMA, L1, L2, NFT, "
                    "биржа, кошелёк, стейкинг, airdrop"
                ),
            )
        full_transcript.append(response.text)
        os.unlink(path)

    return "\n\n".join(full_transcript)


# ==================== ГЕНЕРАЦИЯ РЕЗЮМЕ ====================

def format_duration(seconds):
    """Форматируем длительность в читаемый вид"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours} ч {minutes} мин"
    return f"{minutes} мин"


def generate_summary(transcript, duration_str):
    """Генерируем резюме через GPT-4o-mini"""
    prompt = SUMMARY_PROMPT.format(transcript=transcript, duration=duration_str)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4000,
    )
    return response.choices[0].message.content


# ==================== GOOGLE DOCS ====================

def get_google_docs_service():
    """Подключаемся к Google Docs API"""
    creds_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/documents"],
    )
    return build("docs", "v1", credentials=creds)


def append_to_google_doc(summary_text, meeting_date):
    """Дописываем резюме в конец Google Doc с форматированием"""
    service = get_google_docs_service()

    doc = service.documents().get(documentId=GOOGLE_DOC_ID).execute()
    end_index = doc["body"]["content"][-1]["endIndex"] - 1

    separator = "\n\n" + "=" * 60 + "\n"
    full_text = f"{separator}{meeting_date}\n\n{summary_text}\n"

    # Вставляем текст
    service.documents().batchUpdate(
        documentId=GOOGLE_DOC_ID,
        body={"requests": [{"insertText": {"location": {"index": end_index}, "text": full_text}}]},
    ).execute()

    # Форматируем дату красным + жирным
    date_start = end_index + len(separator)
    date_end = date_start + len(meeting_date)

    first_newline = summary_text.index("\n") if "\n" in summary_text else len(summary_text)

    format_requests = [
        {
            "updateTextStyle": {
                "range": {"startIndex": date_start, "endIndex": date_end},
                "textStyle": {
                    "bold": True,
                    "foregroundColor": {"color": {"rgbColor": {"red": 1.0, "green": 0.0, "blue": 0.0}}},
                },
                "fields": "bold,foregroundColor",
            }
        },
        {
            "updateTextStyle": {
                "range": {"startIndex": date_end + 2, "endIndex": date_end + 2 + first_newline},
                "textStyle": {"bold": True},
                "fields": "bold",
            }
        },
    ]

    service.documents().batchUpdate(
        documentId=GOOGLE_DOC_ID,
        body={"requests": format_requests},
    ).execute()


# ==================== ОБРАБОТКА АУДИО ====================

def process_audio(chat_id, file_id):
    """Основной пайплайн: скачать → транскрибировать → резюме → Google Doc"""
    try:
        send_message(chat_id, "⏳ Скачиваю файл...")
        audio_path = download_telegram_file(file_id)

        send_message(chat_id, "✂️ Обрабатываю аудио...")
        chunks, duration_sec = split_audio(audio_path)
        os.unlink(audio_path)
        duration_str = format_duration(duration_sec)

        send_message(chat_id, f"🎙 Транскрибирую ({len(chunks)} частей, ~{duration_str})...")
        transcript = transcribe_audio(chunks)

        send_message(chat_id, "📝 Генерирую резюме...")
        summary = generate_summary(transcript, duration_str)

        # Дата
        from datetime import datetime
        months_ru = [
            "", "января", "февраля", "марта", "апреля", "мая", "июня",
            "июля", "августа", "сентября", "октября", "ноября", "декабря",
        ]
        now = datetime.now()
        meeting_date = f"{now.day} {months_ru[now.month]} {now.year}"

        send_message(chat_id, "📄 Записываю в Google Doc...")
        append_to_google_doc(summary, meeting_date)

        doc_url = f"https://docs.google.com/document/d/{GOOGLE_DOC_ID}/edit"
        send_message(chat_id, f"✅ Готово!\n\n📋 {summary[:500]}...\n\n📄 Документ: {doc_url}")

    except Exception as e:
        send_message(chat_id, f"❌ Ошибка: {str(e)}")
        print(f"Error: {e}")


# ==================== ВЕБХУК TELEGRAM ====================

@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    """Обработчик вебхука от Telegram"""
    data = request.json
    if not data or "message" not in data:
        return jsonify({"ok": True})

    message = data["message"]
    chat_id = str(message["chat"]["id"])

    # Проверяем, что сообщение от авторизованного пользователя
    if chat_id != TELEGRAM_CHAT_ID:
        send_message(chat_id, "⛔ У тебя нет доступа к этому боту.")
        return jsonify({"ok": True})

    # Команда /start
    if "text" in message and message["text"].startswith("/start"):
        send_message(chat_id, (
            "👋 Привет! Я бот для создания резюме встреч.\n\n"
            "Отправь мне аудио- или видеофайл записи встречи, "
            "и я создам резюме в Google Doc.\n\n"
            "Поддерживаемые форматы: mp3, mp4, m4a, ogg, wav, webm"
        ))
        return jsonify({"ok": True})

    # Обработка аудио/видео/документа
    file_id = None

    if "audio" in message:
        file_id = message["audio"]["file_id"]
    elif "voice" in message:
        file_id = message["voice"]["file_id"]
    elif "video" in message:
        file_id = message["video"]["file_id"]
    elif "video_note" in message:
        file_id = message["video_note"]["file_id"]
    elif "document" in message:
        mime = message["document"].get("mime_type", "")
        if mime.startswith(("audio/", "video/")) or mime == "application/octet-stream":
            file_id = message["document"]["file_id"]
        else:
            # Проверяем по расширению
            fname = message["document"].get("file_name", "")
            valid_ext = (".mp3", ".mp4", ".m4a", ".ogg", ".wav", ".webm", ".oga", ".flac")
            if fname.lower().endswith(valid_ext):
                file_id = message["document"]["file_id"]
            else:
                send_message(chat_id, "⚠️ Отправь аудио- или видеофайл записи встречи.")
                return jsonify({"ok": True})

    if file_id:
        # Запускаем обработку в отдельном потоке, чтобы Telegram не ждал
        import threading
        thread = threading.Thread(target=process_audio, args=(chat_id, file_id))
        thread.start()
        return jsonify({"ok": True})

    send_message(chat_id, "⚠️ Отправь аудио- или видеофайл записи встречи.")
    return jsonify({"ok": True})


# ==================== НАСТРОЙКА ВЕБХУКА ====================

@app.route("/set_webhook", methods=["GET"])
def set_webhook():
    """Вызови этот URL один раз для регистрации вебхука Telegram"""
    # Railway даёт домен через переменную RAILWAY_PUBLIC_DOMAIN
    domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
    if not domain:
        return jsonify({"error": "RAILWAY_PUBLIC_DOMAIN not set"}), 400

    webhook_url = f"https://{domain}/webhook"
    resp = requests.post(f"{TELEGRAM_API}/setWebhook", json={"url": webhook_url})
    return jsonify(resp.json())


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Meeting Summary Telegram Bot"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
