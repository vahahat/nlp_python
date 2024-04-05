from transformers import BertTokenizer, BertModel
import torch

# Инициализация токенизатора и модели
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Текст для анализа
text = "Hello, how are you doing today?"

# Токенизация текста
input_ids = tokenizer.encode(text, add_special_tokens=True)
input_tensor = torch.tensor([input_ids])

# Получение выходных данных модели
with torch.no_grad():
    outputs = model(input_tensor)
    # Последнее скрытое состояние
    last_hidden_states = outputs.last_hidden_state

# `last_hidden_states` содержит векторные представления для каждого слова в предложении
