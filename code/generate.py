import torch
from model import RNNAttentionLanguageModel, Attention
from utils import decode

# Предполагаем, что у вас есть экземпляр модели 'model'
# и стартовая последовательность 'start_tokens' (например, начальные символы из вашего текста)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Генерация дополнительных токенов
max_new_tokens = 500  # Задайте желаемое количество генерируемых токенов
generated_tokens = model.generate(context, max_new_tokens, temperature=0.6)

# Декодирование сгенерированной последовательности в текст
generated_text = decode(generated_tokens[0].tolist())
print(generated_text)
