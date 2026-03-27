import os
import re
import json
import math
import tempfile
import threading
import subprocess
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from openai import OpenAI
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

app = Flask(__name__)

# Устанавливаем ffmpeg при старте если не установлен
if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
    print("ffmpeg не найден, устанавливаем...")
    subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)
    print("ffmpeg установлен")

# ==================== НАСТРОЙКИ ====================

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
GOOGLE_DOC_ID = os.environ["GOOGLE_DOC_ID"]
GOOGLE_CREDENTIALS_JSON = os.environ["GOOGLE_CREDENTIALS_JSON"]

# Максимальный размер куска для Whisper API (МБ)
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
    """Скачать файл из Telegram во временный файл (лимит 20 МБ)"""
    resp = requests.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id})
    data = resp.json()

    if not data.get("ok"):
        raise ValueError(
            "Файл слишком большой (лимит Telegram — 20 МБ).\n\n"
            "Загрузи файл на Google Drive с открытым доступом и отправь ссылку командой:\n"
            "/url https://drive.google.com/file/d/XXXXX/view"
        )

    file_path = data["result"]["file_path"]
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    response = requests.get(file_url, stream=True)

    ext = os.path.splitext(file_path)[1] or ".ogg"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    for chunk in response.iter_content(chunk_size=8192):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


def download_file_from_url(url):
    """Скачать файл по прямой ссылке (поддерживает Google Drive)"""
    file_id = None

    # Конвертируем Google Drive ссылку в прямую ссылку скачивания
    gdrive_match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if not gdrive_match:
        gdrive_match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)

    if gdrive_match:
        file_id = gdrive_match.group(1)
        url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"

    session = requests.Session()
    response = session.get(url, stream=True, allow_redirects=True)

    # Google Drive для больших файлов показывает страницу подтверждения — обходим
    if "text/html" in response.headers.get("Content-Type", ""):
        confirm_match = re.search(r'confirm=([0-9A-Za-z_-]+)', response.text)
        if confirm_match and file_id:
            confirm_token = confirm_match.group(1)
            response = session.get(
                f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}",
                stream=True
            )
        else:
            raise ValueError(
                "Не удалось скачать файл с Google Drive.\n"
                "Убедись, что доступ к файлу открыт: Настройки → Доступ → Все, у кого есть ссылка."
            )

    # Определяем расширение из Content-Disposition или из URL
    content_disposition = response.headers.get("Content-Disposition", "")
    ext_match = re.search(r'filename[^;=\n]*=.*?\.([a-z0-9]+)["\s]', content_disposition, re.IGNORECASE)
    if ext_match:
        ext = f".{ext_match.group(1)}"
    else:
        ext = os.path.splitext(url.split("?")[0])[1] or ".m4a"

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    for chunk in response.iter_content(chunk_size=8192):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


# ==================== НАРЕЗКА АУДИО (ffmpeg) ====================

def get_audio_duration(file_path):
    """Получаем длительность аудио через ffprobe (без загрузки в память)"""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def split_audio(file_path):
    """
    Нарезаем аудио на куски по ~24 МБ для Whisper API.
    Используем ffmpeg напрямую — файл не загружается в память целиком.
    """
    duration_seconds = get_audio_duration(file_path)

    # Конвертируем в mono mp3 32k и смотрим итоговый размер
    tmp_probe = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_probe.close()
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", file_path,
            "-ac", "1", "-ar", "16000", "-b:a", "32k",
            tmp_probe.name,
        ],
        capture_output=True
    )
    file_size_mb = os.path.getsize(tmp_probe.name) / (1024 * 1024)
    os.unlink(tmp_probe.name)

    if file_size_mb <= MAX_CHUNK_SIZE_MB:
        # Файл влезает целиком — просто конвертируем
        tmp_single = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_single.close()
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", file_path,
                "-ac", "1", "-ar", "16000", "-b:a", "32k",
                tmp_single.name,
            ],
            capture_output=True
        )
        return [tmp_single.name], duration_seconds

    # Нарезаем на куски по времени
    num_chunks = math.ceil(file_size_mb / MAX_CHUNK_SIZE_MB)
    chunk_duration = duration_seconds / num_chunks

    chunk_files = []
    for i in range(num_chunks):
        start = i * chunk_duration
        tmp_chunk = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_chunk.close()
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-t", str(chunk_duration),
                "-i", file_path,
                "-ac", "1", "-ar", "16000", "-b:a", "32k",
                tmp_chunk.name,
            ],
            capture_output=True
        )
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
    """Вставляем резюме в НАЧАЛО Google Doc (после заголовка) с форматированием"""
    service = get_google_docs_service()
    doc = service.documents().get(documentId=GOOGLE_DOC_ID).execute()

    content = doc["body"]["content"]
    if len(content) > 2:
        insert_index = content[2]["startIndex"]
    else:
        insert_index = content[-1]["endIndex"] - 1

    separator = "=" * 60 + "\n"
    full_text = f"{separator}{meeting_date}\n\n{summary_text}\n\n"

    service.documents().batchUpdate(
        documentId=GOOGLE_DOC_ID,
        body={"requests": [{"insertText": {"location": {"index": insert_index}, "text": full_text}}]},
    ).execute()

    date_start = insert_index + len(separator)
    date_end = date_start + len(meeting_date)
    first_newline = summary_text.index("\n") if "\n" in summary_text else len(summary_text)

    service.documents().batchUpdate(
        documentId=GOOGLE_DOC_ID,
        body={"requests": [
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
        ]},
    ).execute()


# ==================== ОБЩИЙ ПАЙПЛАЙН ====================

def run_pipeline(chat_id, audio_path):
    """Общий пайплайн: аудиофайл → транскрипция → резюме → Google Doc"""
    try:
        send_message(chat_id, "✂️ Обрабатываю аудио...")
        chunks, duration_sec = split_audio(audio_path)
        os.unlink(audio_path)
        duration_str = format_duration(duration_sec)

        send_message(chat_id, f"🎙 Транскрибирую ({len(chunks)} частей, ~{duration_str})...")
        transcript = transcribe_audio(chunks)

        send_message(chat_id, "📝 Генерирую резюме...")
        summary = generate_summary(transcript, duration_str)

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


def process_audio(chat_id, file_id):
    """Скачать файл из Telegram и запустить пайплайн"""
    try:
        send_message(chat_id, "⏳ Скачиваю файл...")
        audio_path = download_telegram_file(file_id)
    except Exception as e:
        send_message(chat_id, f"❌ Ошибка: {str(e)}")
        print(f"Error: {e}")
        return
    run_pipeline(chat_id, audio_path)


def process_from_url(chat_id, url):
    """Скачать файл по ссылке и запустить пайплайн"""
    try:
        send_message(chat_id, "⏳ Скачиваю файл по ссылке...")
        audio_path = download_file_from_url(url)
    except Exception as e:
        send_message(chat_id, f"❌ Ошибка: {str(e)}")
        print(f"Error: {e}")
        return
    run_pipeline(chat_id, audio_path)


# ==================== ВЕБХУК TELEGRAM ====================

@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    """Обработчик вебхука от Telegram"""
    data = request.json
    if not data or "message" not in data:
        return jsonify({"ok": True})

    message = data["message"]
    chat_id = str(message["chat"]["id"])

    if chat_id != TELEGRAM_CHAT_ID:
        send_message(chat_id, "⛔ У тебя нет доступа к этому боту.")
        return jsonify({"ok": True})

    # Команда /start
    if "text" in message and message["text"].startswith("/start"):
        send_message(chat_id, (
            "👋 Привет! Я бот для создания резюме встреч.\n\n"
            "Отправь мне аудио- или видеофайл записи встречи (до 20 МБ), "
            "и я создам резюме в Google Doc.\n\n"
            "Для больших файлов загрузи запись на Google Drive и отправь ссылку:\n"
            "/url https://drive.google.com/file/d/XXXXX/view\n\n"
            "Поддерживаемые форматы: mp3, mp4, m4a, ogg, wav, webm"
        ))
        return jsonify({"ok": True})

    # Команда /url или голая ссылка на Google Drive
    text = message.get("text", "") or message.get("caption", "")
    is_url_command = text.startswith("/url")
    is_bare_gdrive = (
        not text.startswith("/") and
        re.search(r'https?://drive\.google\.com/\S+', text)
    )

    if is_url_command or is_bare_gdrive:
        if is_url_command:
            parts = text.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip().startswith("http"):
                send_message(chat_id, (
                    "⚠️ Укажи ссылку после команды.\n\n"
                    "Пример:\n/url https://drive.google.com/file/d/XXXXX/view"
                ))
                return jsonify({"ok": True})
            url = parts[1].strip()
        else:
            match = re.search(r'https?://drive\.google\.com/\S+', text)
            url = match.group(0).rstrip(")")

        thread = threading.Thread(target=process_from_url, args=(chat_id, url))
        thread.start()
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
            fname = message["document"].get("file_name", "")
            valid_ext = (".mp3", ".mp4", ".m4a", ".ogg", ".wav", ".webm", ".oga", ".flac")
            if fname.lower().endswith(valid_ext):
                file_id = message["document"]["file_id"]
            else:
                send_message(chat_id, "⚠️ Отправь аудио- или видеофайл записи встречи.")
                return jsonify({"ok": True})

    if file_id:
        thread = threading.Thread(target=process_audio, args=(chat_id, file_id))
        thread.start()
        return jsonify({"ok": True})

    send_message(chat_id, "⚠️ Отправь аудио- или видеофайл записи встречи.")
    return jsonify({"ok": True})


# ==================== НАСТРОЙКА ВЕБХУКА ====================

@app.route("/set_webhook", methods=["GET"])
def set_webhook():
    """Вызови этот URL один раз для регистрации вебхука Telegram"""
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
