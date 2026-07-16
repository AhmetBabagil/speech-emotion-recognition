#bu dosyanın görevi veri setlerindeki ses kayıtlarını mlp modelininin kullanabilceği sayısal özelliklere dönüştürmektedir her ses kaydının librosa kullanılaraka mel-spectrogram öujarukacaj ve soejtıfram sebit 64x64 boyutuna getirikecek ve düzleştrilerek 4096 boyutunda bir vektör elde edilecekeitrç hesağlanan vektörler sonraki model denemelerinde tekrar hesaplanmamaları için cache klasöreün kaydedilecektir

import hashlib
from pathlib import Path
import numpy as np
import librosa

SAMPLE_RATE = 16000
N_MELS = 64
N_FRAMES = 64
N_FFT = 1024
HOP_LENGTH = 512
VECTOR_SIZE = N_MELS * N_FRAMES


def extract_melspec(audio_path):
      # Ses dosyasını 4096 boyutlu bir özellik vektörüne dönüştürme işlemi yipaıcalkscıtr bu ksımda

      audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

      mel=librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
             n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
      )
      #şimdi enerji değelrini desibebl ölceğine çevirelim : 
      mel_db = librosa.power_to_db(mel, ref=np.max)
      current_frames = mel_db.shape[1]
      if current_frames < N_FRAMES:
              pad_amount = N_FRAMES - current_frames
              mel_fixed = np.pad(
      mel_db,
      ((0, 0), (0, pad_amount)),
      mode="constant",
      constant_values=mel_db.min(),
  )
      else:
              start = (current_frames - N_FRAMES) // 2
              mel_fixed = mel_db[:, start:start +
              N_FRAMES]
      return mel_fixed.reshape(VECTOR_SIZE).astype(np.float32)
     # dönüştürecek.

