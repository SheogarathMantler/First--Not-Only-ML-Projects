import numpy as np
import simpleaudio as sa
import matplotlib.pyplot as plt

def filtered(x, type, **kwargs):
    res = None
    if type == 'Linear':
        alpha = kwargs.get('a')
        res = x*alpha
    # TODO Add other types
    return res

def loop(iter, excitation):
    n = len(excitation)
    res = excitation.copy()
    for i in range(iter):
        y =filtered(res[-n], 'Linear', a=0.5)
        res = np.append(res, y)
    return res


frequency = 440                                                  # Наша сыгранная нота будет 440 Гц
fs = 44100                                                       # 44100 выборок в секунду
exc_seconds = 0.1                                                # Длительность возбуждающего сигнала - 0.1 сек
duration = 3                                                     # длительность всего звука
t = np.linspace(0, exc_seconds, int(exc_seconds * fs), False)    # Генерируем массив с секундами * сэмплированием шагов, в диапазоне от 0 до 0.1 секунд
excitation = np.sin(frequency * t * 2 * np.pi)                         # Генерация синусоидальной волны 440 Гц
note_echoed = loop(int((duration-exc_seconds) * fs), excitation)
audio = note_echoed * (2 ** 15 - 1) / np.max(np.abs(note_echoed))# максимальное значение находится в 16-битном диапазоне
print(note_echoed.shape, excitation.shape)
audio = audio.astype(np.int16)                                   # Конвертировать в 16-битные данные
play_obj = sa.play_buffer(audio, 1, 2, fs)                       # Начать воспроизведение
play_obj.wait_done()                                             # Дождитесь окончания воспроизведения перед выходом



