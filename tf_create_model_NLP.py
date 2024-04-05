from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Предположим, что X_train - это ваш подготовленный набор данных, а y_train - метки классов
# Для примера, допустим, что ваши данные уже предобработаны и векторизованы

max_features = 10000  # Количество уникальных слов
maxlen = 500  # Максимальная длина текста
embedding_size = 128

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# Обучение модели
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)
