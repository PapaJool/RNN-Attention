import os
import torch
from model import RNNAttentionLanguageModel, Attention
from utils import decode

# Получаем путь к текущей директории скрипта
current_dir = os.path.dirname(os.path.abspath(__file__))

# Загрузка модели
model = RNNAttentionLanguageModel()
model.load_state_dict(torch.load(os.path.join(current_dir, 'checkpoints', 'your_model.pth')))
model.eval()

# Получаем путь к файлу input.txt в папке data
input_file_path = os.path.join(current_dir, 'data', 'input.txt')

# Загрузка стартовой последовательности из файла input.txt
with open(input_file_path, 'r', encoding='utf-8') as f:
    start_tokens = torch.tensor([int(token) for token in f.read().split()], dtype=torch.long).unsqueeze(0)

# Генерация дополнительных токенов
max_new_tokens = 500  # Задайте желаемое количество генерируемых токенов
temperature = 0.6  # Задайте желаемую температуру

generated_tokens = model.generate(start_tokens, max_new_tokens, temperature)

# Декодирование сгенерированной последовательности в текст
generated_text = decode(generated_tokens[0].tolist())
print(generated_text)
