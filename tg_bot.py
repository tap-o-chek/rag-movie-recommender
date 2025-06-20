from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, CallbackQueryHandler, filters, ContextTypes
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from langdetect import detect
import asyncio
from telegram import BotCommand
import faiss
import numpy as np
import pandas as pd


df = pd.read_pickle(r"C:\my_projects\rag-movie-recommender\movies_with_embeddings.pkl")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# FAISS-индекс
d = len(df['embedding'][0])
index = faiss.IndexFlatL2(d)
index.add(np.vstack(df['embedding'].values))

def search_similar_movies(query, shown_indices=set(), top_k=5):
    lang = detect(query)

    # Перевод запроса на английский, если он не на английском
    if lang != 'en':
        try:
            translated_query = GoogleTranslator(source='auto', target='en').translate(query)
        except Exception:
            translated_query = query
    else:
        translated_query = query

    query_vector = model.encode([translated_query])
    distances, indices = index.search(query_vector, top_k + len(shown_indices))

    results = []
    new_indices = []

    for idx, dist in zip(indices[0], distances[0]):
        if idx in shown_indices:
            continue
        row = df.iloc[idx]
        overview = row['overview']
        
        # перевод описания на язык пользователя
        if lang != 'en':
            try:
                overview = GoogleTranslator(source='en', target=lang).translate(overview)
            except Exception:
                pass

        results.append(
            f"🎬 *{row['title']}*\n{overview}\n(Сходство: {1 - dist:.2f})\n"
        )
        new_indices.append(idx)
        if len(results) >= top_k:
            break

    return "\n".join(results), new_indices

# при /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! 👋 Я — рекомендательный бот фильмов на основе ИИ.\n\n"
        "Просто напиши, что тебе интересно — жанр, описание или настроение, например:\n"
        "🔍 *детектив в снегах*\n"
        "🔍 *комедия с путешествиями во времени*\n"
        "🔍 *роботы и космос*\n\n"
        "А я подберу для тебя фильмы и объясню, почему они подходят 🎬",
        parse_mode='Markdown'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    await update.message.reply_text("🔎 Подбираю фильмы, пожалуйста, подождите...")

    context.user_data['query'] = query
    context.user_data['shown_indices'] = []

    answer, new_indices = search_similar_movies(query, shown_indices=set())
    context.user_data['shown_indices'].extend(new_indices)

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Ещё 5 фильмов", callback_data="more")]
    ])
    await update.message.reply_text(answer, parse_mode="Markdown", reply_markup=keyboard)

# кнопка "Ещё"
# 🔁 Обработка кнопки "Ещё"
async def more_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = context.user_data.get('query')
    shown = set(context.user_data.get('shown_indices', []))

    if not query:
        await update.callback_query.answer("Сначала введите запрос.")
        return

    await update.callback_query.answer("🔄 Подбираю ещё...")

    answer, new_indices = search_similar_movies(query, shown_indices=shown)
    context.user_data['shown_indices'].extend(new_indices)

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Ещё 5 фильмов", callback_data="more")]
    ])
    
    # ⬅️ ВАЖНО: используем send_message вместо edit_message_text
    await update.effective_chat.send_message(answer, parse_mode="Markdown", reply_markup=keyboard)



if __name__ == '__main__':
    app = ApplicationBuilder().token("7909924378:AAHmguToHsEyRAJS5qihCCDV91Ww8-RxMjE").build()

    # меню команд 
    async def setup_commands():
        commands = [
            BotCommand("start", "Начать/перезапустить работу с ботом"),\
        ]
        await app.bot.set_my_commands(commands)

    # установка команд
    asyncio.get_event_loop().run_until_complete(setup_commands())

    # обработчики
    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(more_callback, pattern="more"))

    print("Бот запущен")
    app.run_polling()