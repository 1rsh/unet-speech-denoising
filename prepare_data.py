import os
import librosa
import numpy as np
from tqdm.auto import tqdm
import soundfile as sf
import skimage.io
import splitfolders

def CreateData(speech_dir, noise_dir, sample_rate, frame_length, min_duration, nb_samples, hop_length, path_save_sound):
    
    noise = audio_to_npy(noise_dir, sample_rate, frame_length, min_duration)
    voice = audio_to_npy(speech_dir, sample_rate, frame_length, min_duration)

    
    prod_voice, prod_noise, prod_noisy_voice = blend_noise_randomly(voice, noise, nb_samples, frame_length)
    print()
    for i in tqdm(range(len(prod_voice)), desc = f"Saving to {path_save_sound}"):

        noisy_voice_long = prod_noisy_voice[i]
        sf.write(path_save_sound + 'noisy_voice/'+str(i)+'.wav', noisy_voice_long[:], sample_rate)
        spectrogram_image(hop_length, noisy_voice_long, "noisy_voice/"+str(i)+".png")
        voice_long = prod_voice[i]
        sf.write(path_save_sound + 'voice/'+str(i)+'.wav', voice_long[:], sample_rate)
        spectrogram_image(hop_length, voice_long, "voice/"+str(i)+".png")        
        noise_long = prod_noise[i]
        sf.write(path_save_sound + 'noise/'+str(i)+'.wav', noise_long[:], sample_rate)
        spectrogram_image(hop_length, noise_long, "noise/"+str(i)+".png")


def audio_chunker(sound_data, frame_length):
    sequence_sample_length = sound_data.shape[0]

    sound_data_list = [sound_data[start:start + frame_length] for start in range(
    0, sequence_sample_length - frame_length + 1, frame_length + 1)]  # get sliding windows
    if(len(sound_data_list)):
        sound_data_array = np.vstack(sound_data_list)
        return sound_data_array
    

def audio_to_npy(audio_dir, sample_rate, frame_length, min_duration):
    list_sound_arr = []

    for file in tqdm(os.listdir(audio_dir)[:10000], desc = f"Loading from {audio_dir}"):
        if file.endswith(".wav") or file.endswith(".3gp"):
            y, sr = librosa.load(os.path.join(audio_dir, file), sr = sample_rate)
            total_dur = librosa.get_duration(y = y, sr = sr)

            if total_dur >= min_duration:
                alp = audio_chunker(y, frame_length)
                try:
                    if len(alp[0]) == frame_length:
                        list_sound_arr.append(alp)
                except:
                    continue


    return np.vstack(list_sound_arr)



def blend_noise_randomly(voice, noise, nb_samples, frame_length):
    prod_voice = np.zeros((nb_samples, frame_length))
    prod_noise = np.zeros((nb_samples, frame_length))
    prod_noisy_voice = np.zeros((nb_samples, frame_length))

    for i in range(nb_samples):
        id_voice = np.random.randint(0, voice.shape[0])
        id_noise = np.random.randint(0, noise.shape[0])
        level_noise = np.random.uniform(0.2, 0.8)
        prod_voice[i, :] = voice[id_voice, :]
        prod_noise[i, :] = level_noise * noise[id_noise, :]
        prod_noisy_voice[i, :] = prod_voice[i, :] + prod_noise[i, :]

    return prod_voice, prod_noise, prod_noisy_voice

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spectrogram_image(hop_length_fft, audio, save_loc):

    spectrogram = librosa.stft(audio, n_fft=2*hop_length_fft, hop_length=hop_length_fft)
    
    spectrogram = librosa.amplitude_to_db(
        np.abs(spectrogram), ref=np.max)

    spectrogram = scale_minmax(spectrogram, 0, 255).astype(np.uint8)
    spectrogram = np.flip(spectrogram, axis=0) 
    spectrogram = 255-spectrogram
    skimage.io.imsave("data/processed/spectrogram/"+save_loc, spectrogram)
    


CreateData("data/speech-data", "data/noise-data/audio", 8000, 5*8000, 5, 1000, 512, "data/processed/audio/")
