from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from langdetect import detect
import faiss
import numpy as np
import pandas as pd


df = pd.read_pickle(r"C:\my_projects\rag-movie-recommender\movies_with_embeddings.pkl")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# FAISS-индекс
d = len(df['embedding'][0])
index = faiss.IndexFlatL2(d)
index.add(np.vstack(df['embedding'].values))

# перевод описания фильма
def translate_text(text, target='ru'):
    try:
        return GoogleTranslator(source='auto', target=target).translate(text)
    except Exception as e:
        return text  # если ошибка, возвращаем оригинал

# поисковая функция
def search_similar_movies(query, top_k=5):
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    
    try:
        query_lang = detect(query)
    except:
        query_lang = 'en'
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        row = df.iloc[idx]
        title = row['title']
        overview = row['overview']
        if query_lang == 'ru':
            overview = translate_text(overview, target='ru')
        results.append(f"🎬 *{title}*\n{overview}\n(Сходство: {1 - dist:.2f})\n")
    return "\n".join(results)

# обработчик сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    waiting_msg = await update.message.reply_text("🔎 Подбираю фильмы, пожалуйста, подождите...")
    answer = search_similar_movies(query)
    
    await waiting_msg.delete()
    await update.message.reply_text(answer, parse_mode="Markdown")


# приветственное сообщение при /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я — рекомендательный бот фильмов.\n\n"
        "Просто напиши, что тебе интересно — жанр, описание или настроение, например:\n"
        "🔍 *детектив в снегах*\n"
        "🔍 *комедия с путешествиями во времени*\n"
        "🔍 *роботы и космос*\n\n"
        "А я подберу для тебя фильмы и расскажу, почему они подходят)",
                parse_mode='Markdown'
    )

if __name__ == '__main__':
    app = ApplicationBuilder().token("7909924378:AAHmguToHsEyRAJS5qihCCDV91Ww8-RxMjE").build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print('Бот запущен')
    app.run_polling()