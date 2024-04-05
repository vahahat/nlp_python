import tensorflow_hub as hub
import numpy as np
import tensorflow_text  # Необходим для операций предобработки текста

# Загрузка модели USE
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Тексты для преобразования
sentences = ["This is a sentence.", "This is another sentence."]

# Получение векторных представлений
embeddings = embed(sentences)

# `embeddings` содержит векторные представления предложений
for i, embedding in enumerate(np.array(embeddings)):
    print(f"Sentence: {sentences[i]}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding: {embedding[:5]}")
    print("---")
