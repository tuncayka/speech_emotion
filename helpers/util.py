import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa
from librosa import display
import soundfile

import matplotlib.pyplot as plt


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


def load_data(filepath):
    return librosa.load(filepath)

def plot_waveplot(data, sr):
    display.waveplot(data, sr=sr)
    plt.show()
    
def show_spectrogram(spectrogram, figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.xlabel('time')
    plt.ylabel('freqs')
    plt.imshow(spectrogram)    
    
def plot_compare(files = []):
    assert len(files)>1 
    len_files = len(files)
    wave_axes = []
    spec_axes = []
    mel_axes = []
    chroma_axes = []
    mfcc_axes = []
    
    fig = plt.figure(figsize=(5 * len_files, 6))
    for f in files:
        data, sample_rate = librosa.load(f)
#         sample_rate, samples = wavfile.read(f)
#         freqs, times, spectrogram = log_specgram(samples, sample_rate)   
        
        ax = fig.add_subplot(len_files, 1, len(wave_axes)+1)
        wave_axes.append(ax)
        ax.set_title('All wave of {}'.format(f))
        ax.set_ylabel('Amplitude')
        librosa.display.waveplot(data, sample_rate, ax=ax)
    plt.tight_layout()
    
    fig = plt.figure(figsize=(5 * len_files, 6))
    for f in files:
#         data, sample_rate = librosa.load(f)
        sample_rate, samples = wavfile.read(f)
        freqs, times, spectrogram = log_specgram(samples, sample_rate) 
        
        ax = fig.add_subplot(len_files, 1, len(spec_axes)+1)
        ax.imshow(spectrogram.T, aspect='auto', origin='lower', 
                   extent=[times.min(), times.max(), freqs.min(), freqs.max()])
        spec_axes.append(ax)
        ax.set_yticks(freqs[::16])
        ax.set_xticks(times[::16])
        ax.set_title('Spectrogram of ' + f)
        ax.set_ylabel('Freqs in Hz')
        ax.set_xlabel('Seconds')   
    plt.tight_layout()        
        
    fig = plt.figure(figsize=(5 * len_files, 6))
    for f in files:        
        data, sample_rate = librosa.load(f)
        S = librosa.feature.melspectrogram(data, sr=sample_rate, n_mels=128)

        ax = fig.add_subplot(len_files, 1, len(mel_axes)+1)
        # Convert to log scale (dB). We'll use the peak power (max) as reference.
        log_S = librosa.power_to_db(S, ref=np.max)

        librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel', ax=ax)
        mel_axes.append(ax)
        ax.set_title('Mel power spectrogram')
#         plt.colorbar(format='%+02.0f dB', ax=ax)  
    plt.tight_layout()        
        
    fig = plt.figure(figsize=(5 * len_files, 6))
    for f in files:        
        data, sample_rate = librosa.load(f)
        c = librosa.feature.chroma_stft(data, sr=sample_rate)
        
        ax = fig.add_subplot(len_files, 1, len(chroma_axes)+1)
        log_c = librosa.power_to_db(c, ref=np.max)

        librosa.display.specshow(log_c, sr=sample_rate, x_axis='time', y_axis='chroma', ax=ax)
        chroma_axes.append(ax)
        ax.set_title("Chroma Graph")
#         plt.colorbar(format='%+02.0f dB', ax=ax)  
    plt.tight_layout()   
    
    fig = plt.figure(figsize=(5 * len_files, 6))
    for f in files:        
        data, sample_rate = librosa.load(f)
        m = librosa.feature.mfcc(data, sr=sample_rate)
        
        ax = fig.add_subplot(len_files, 1, len(mfcc_axes)+1)
        log_m = librosa.power_to_db(m, ref=np.max)

        librosa.display.specshow(log_m, sr=sample_rate, x_axis='time', y_axis='chroma', ax=ax)
        mfcc_axes.append(ax)
        ax.set_title("MFCC Graph")
#         plt.colorbar(format='%+02.0f dB', ax=ax)  
    plt.tight_layout()        
        
    plt.tight_layout()
    plt.show()
        
    
def plot_all_together(file_name):
    data, sample_rate = librosa.load(file_name)
    sample_rate, samples = wavfile.read(file_name)
    freqs, times, spectrogram = log_specgram(samples, sample_rate)

    fig = plt.figure(figsize=(14, 8))
    ax0 = fig.add_subplot(311)
    ax0.set_title('All wave of {}'.format(file_name))
    ax0.set_ylabel('Amplitude')
    librosa.display.waveplot(data, sample_rate, ax=ax0)

    ax1 = fig.add_subplot(312)
    ax1.set_title('Raw wave of ' + file_name)
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples[:sample_rate])

    ax2 = fig.add_subplot(313)
    ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    ax2.set_yticks(freqs[::16])
    ax2.set_xticks(times[::16])
    ax2.set_title('Spectrogram of ' + file_name)
    ax2.set_ylabel('Freqs in Hz')
    ax2.set_xlabel('Seconds')
    plt.tight_layout()
    
    fig = plt.figure(figsize=(14, 12))
    ax3 = fig.add_subplot(311)
    S = librosa.feature.melspectrogram(data, sr=sample_rate, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    
    librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
    ax3.set_title('Mel power spectrogram ')
    plt.colorbar(format='%+02.0f dB', ax=ax3)
    
    ax4 = fig.add_subplot(312)
    
    c = librosa.feature.chroma_stft(data, sr=sample_rate)

    log_c = librosa.power_to_db(c, ref=np.max)

    librosa.display.specshow(log_c, sr=sample_rate, x_axis='time', y_axis='chroma')
    ax4.set_title("Chroma Graph")
    plt.colorbar(format="%+02.0f dB", ax=ax4)  
    
    ax5 = fig.add_subplot(313)
    
    m = librosa.feature.mfcc(data, sr=sample_rate)

    log_m = librosa.power_to_db(m, ref=np.max)

    librosa.display.specshow(log_m, sr=sample_rate, x_axis='time', y_axis='chroma')
    ax5.set_title("MFCC Graph")
    plt.colorbar(format="%+02.0f dB", ax=ax5)    
    
    plt.tight_layout()
    plt.show()
    
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))            
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
            
    return result    

def trim_silent(data, db_threshold=10):
    return librosa.effects.trim(data, top_db=db_threshold )[0]

def plot_train_history(history, title, f1=False):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    rows = 3 if f1 else 2

    epochs = range(len(loss))

    fig = plt.figure(figsize=(10, 12))
    
    ax1= fig.add_subplot(rows, 1, 1)

    ax1.plot(epochs, loss, 'b', label='Training loss')
    ax1.plot(epochs, val_loss, 'r', label='Validation loss')
    ax1.set_title(title)
    plt.legend()

    ax2 = fig.add_subplot(rows, 1, 2)
    ax2.plot(epochs, acc, 'b', label='Training Accuracy')
    ax2.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    ax2.set_title(title)
    ax2.legend()
    
    if f1:
        ax3 = fig.add_subplot(rows, 1, 3)
        ax3.plot(epochs, history.history['f1_macro'], 'b', label='Training F1')
        ax3.plot(epochs, history.history['val_f1_macro'], 'r', label='Validation F1')
        ax3.set_title(title)
        ax3.legend()    
    
    plt.tight_layout()
    plt.show()
    
def data_augmentation(file_name, sr=16000, return_original=False):
    """
    Created for data augmentation. 
    
    param file_name: audio file path
    param sr: sample rate
    return_original: Default is False, it True, function returns original file in last element of list
    
    Creates 4 different augmentation:
        1- Add White Noice
        2- Harmonic Effect
        3- Pitch shift wodn by a tritone
        4- Pitch shift up by 3 quarter-tones
    """
    files = []
    
    file, sr = librosa.load(file_name, sr=sr)
    orig_file= np.copy(file)
    
    #white noise
    file = orig_file + 0.005 * np.random.randn(len(orig_file))
    files.append(file)
    
    #harmonic
    file = librosa.effects.harmonic(orig_file, kernel_size=64)
    files.append(file)
    
    #pitch Shift down by a tritone (six half-steps)
    file = librosa.effects.pitch_shift(orig_file, sr, n_steps=-6)
    files.append(file)
    
    #pitch Shift up by 3 quarter-tones
    file = librosa.effects.pitch_shift(orig_file, sr, n_steps=3, bins_per_octave=24)
    files.append(file)
    
    if return_original:
        files.append(orig_file)
        return files
    else:
        return files
        
        
def read_all_data(data_files):
    """
    param data_files: dataframe for holding all files paths. 
    """
    features = []
    target = []
    emb_cols = []
    
    for file in tqdm(data_files['full_path']):
        try:
            #extract other features from file name
    #         actor_num = data_files.query("full_path==@file")['actor_num'].iloc[0]
    #         modelity = data_files.query("full_path==@file")['modelity'].iloc[0]
    #         # vocal_channel is one value.
    #         # vocal_channel = data_files.query("full_path==@file")['vocal_channel'].iloc[0]    
    #         emotional_intensity = data_files.query("full_path==@file")['emotional_intensity'].iloc[0]   
    #         statement = data_files.query("full_path==@file")['statement'].iloc[0]   
    #         repetition = data_files.query("full_path==@file")['repetition'].iloc[0]   
    #         actor_gender = data_files.query("full_path==@file")['actor_gender'].iloc[0] 

    #         emb_cols.append([actor_num, modelity, emotional_intensity, statement, repetition, actor_gender])

            
            with soundfile.SoundFile(file) as f:
                X = f.read(dtype='float32')
                sample_rate = f.samplerate
                # X = util.trim_silent(X, db_threshold=10)
                # Short-time Fourier transform (STFT)
                stft = np.abs(librosa.stft(X))
                result = None
    #             if which == 'mfccs':
    #                 # Mel-frequency cepstral coefficients (MFCCs)
    #                 result = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    #             elif which == 'chroma':
    #                 # Compute a chromagram from a waveform or power spectrogram.
    #                 result = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    #             elif which == 'mel':
    #                 # Compute a mel-scaled spectrogram.
    #                 result = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

                result = np.hstack((mfccs, chroma))
                result = np.hstack((result, mel))
                features.append(result)
            y = data_files.query("full_path==@file")['emotion'].iloc[0]
            target.append(y)
        except Exception as e:
            print("Exception: {}".format(e))
            print("FileName: {}".format(file))
            pass
            
    return np.array(features), np.array(target)
            
    
# I don't like loops. find another solution to here. 
def read_all_data_aug(data_files):
    features = []
    target = []
    
    for file in tqdm(data_files['full_path']):
        y = data_files.query("full_path==@file")['emotion'].iloc[0]
        sr = 16000
        files = data_augmentation(file ,sr=sr, return_original=True)
        result = None
        
        for f in files:
            f = trim_silent(f, db_threshold=10)
            stft = np.abs(librosa.stft(f))
            mfccs = np.mean(librosa.feature.mfcc(y=f, sr=sr, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y=f, sr=sr).T, axis=0)
            
            result = np.hstack((mfccs, chroma))
            result = np.hstack((result, mel))
            target.append(y) 
        
            features.append(result)
        
    return np.array(features), np.array(target)    
