#Batch Norma Sebelum mfm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Max-Feature-Map (MFM) implementation
class MaxFeatureMap(nn.Module):
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)  # Split the channel dimension into two parts
        return torch.max(x1, x2)

class LCNN(nn.Module):
    def __init__(self):
        super(LCNN, self).__init__()
        
        # Blok convolution pertama
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Channel lebih besar
        self.bn1 = nn.BatchNorm2d(32)
        self.mfm1 = MaxFeatureMap()  # MFM menggantikan LeakyReLU
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Blok convolution kedua
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.mfm2 = MaxFeatureMap()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Blok convolution ketiga
        self.conv3 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.mfm3 = MaxFeatureMap()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(44928, 256)  # Sesuaikan dengan ukuran output dari convolution
        self.mfm_fc1 = MaxFeatureMap()  # Fully connected MFM
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 2)  # 2 kelas: suara manusia dan suara AI

    def forward(self, x):
        x = self.pool1(self.mfm1(self.bn1(self.conv1(x))))
        x = self.pool2(self.mfm2(self.bn2(self.conv2(x))))
        x = self.pool3(self.mfm3(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.mfm_fc1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#Baca Model
model = LCNN()  # Inisialisasi model yang sama dengan struktur awal
criterion = torch.nn.CrossEntropyLoss()
model.load_state_dict(torch.load('LCNN CQT.pth'))  # Load state dictionary

import os
import numpy as np
import librosa
import torch

def preprocess_audio(file_path, segment_duration=20, sampling_rate=16000, n_bins=79, fmin=32.7032, hop_length=512, bins_per_octave=10):
    # Load audio file
    data, fs = librosa.load(file_path, sr=sampling_rate)

    # Normalisasi puncak (nilai absolut maksimum menjadi 1)
    data = data / np.max(np.abs(data))

    # Segmentasi audio
    samples_per_segment = int(segment_duration * sampling_rate)
    start_sample = 0  # Ambil segmen pertama
    end_sample = min(samples_per_segment, len(data))
    segment_data = data[start_sample:end_sample]

    # Tambahkan zero padding jika segmen lebih pendek dari durasi penuh
    if len(segment_data) < samples_per_segment:
        segment_data = np.pad(segment_data, (0, samples_per_segment - len(segment_data)), mode='constant')

    # Hitung CQT
    cqt_data = librosa.cqt(
        segment_data,
        sr=sampling_rate,
        n_bins=n_bins,
        fmin=fmin,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave
    )
    cqt_magnitude = librosa.amplitude_to_db(np.abs(cqt_data), ref=np.max)  # Konversi ke dB

    # Konversi CQT menjadi tensor
    cqt_tensor = torch.tensor(cqt_magnitude, dtype=torch.float32)  # [n_bins, waktu]

    # Ubah dimensi menjadi [1, n_bins, waktu] untuk Conv2d
    cqt_tensor = cqt_tensor.unsqueeze(0)  # Menambahkan dimensi channel: [1, n_bins, waktu]

    return cqt_tensor

def evaluate_single_audio_sample(model, file_path, true_label):
    # Preprocess audio untuk mendapatkan tensor
    audio_tensor = preprocess_audio(file_path)
    
    # Ubah dimensi audio tensor agar sesuai dengan input model
    audio_tensor = audio_tensor.unsqueeze(0)  # Tambahkan batch dimension
    
    # Evaluasi model
    model.eval()
    with torch.no_grad():
        output = model(audio_tensor)
        _, predicted = torch.max(output, 1)
        predicted_label = predicted.item()

    # Bandingkan prediksi dengan label sebenarnya
    is_correct = (predicted_label == true_label)
    return predicted_label, is_correct

# Fungsi utama untuk memproses satu file
def process_audio_file(file_path, model, true_label):
    # Langsung evaluasi model tanpa penyesuaian RMS atau resampling
    print("Deteksi sedang berlangsung...")

    # Evaluasi model
    predicted_label, is_correct = evaluate_single_audio_sample(model, file_path, true_label)

    # Tampilkan hasil prediksi
    if predicted_label == 1:
        print("Ini adalah suara AI.")
    else:
        print("Ini bukan suara AI.")

# Contoh penggunaan
file_path = r"D:\Data\Sekolah\Kuliah\penelitian\deteksi AI\suara manu.mp3"  # Path ke file audio
true_label = 1  # Label sebenarnya untuk sampel ini

# Pastikan model sudah dilatih dan diinisialisasi
# model = <model_anda>

process_audio_file(file_path, model, true_label)
