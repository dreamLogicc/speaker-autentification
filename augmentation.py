import numpy as np
import librosa
import soundfile as sf

def add_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def shift(data, shift_max, shift_direction = 'right'):
    shift = np.random.randint(1000, shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

def stretch(data, rate=1):
    input_length = 22050
    data = librosa.effects.time_stretch(y=data, rate=rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data

def volume(y):
    return y*np.random.uniform(2,3)

def augmentation(y ,sr, num_samples):

    i = 0
    augmented_array = []
    while i < num_samples:
        noise_factor = 0.005
        shif_direction = np.random.choice(['right', 'both'])
        shift_range = 5000

        manipulation = np.random.choice(['noise', 'shift','stretch', 'pitch', 'increase_volume'])

        match manipulation:
            case 'noise':
                augmented = add_noise(y, noise_factor=noise_factor)
            case 'shift':
                augmented = shift(y, shift_direction=shif_direction, shift_max=shift_range)
            case 'stretch':
                augmented = stretch(y, rate=1.1)
            case 'pitch':
                augmented = librosa.effects.pitch_shift(y = y, sr= sr  , n_steps=np.random.randint(1,4), bins_per_octave=24)
            case 'increase_volume':
                augmented = volume(y)

        augmented_array.append((augmented, sr))
        i += 1
    return augmented_array

if __name__ == '__main__':

    y ,sr = librosa.load('./fastapi_app/voice_data/Степа/Гайволя Степан Вход (1).wav', mono=True, duration=1)

    noised = add_noise(y, 0.005)
    sf.write('./fastapi_app/test_aug/noised.wav', noised, sr)

    shifted = shift(y, 3000, 'both')
    sf.write('./fastapi_app/test_aug/shifted.wav', shifted, sr)

    stretched = stretch(y, 1.1)
    sf.write('./fastapi_app/test_aug/stretched.wav', stretched, sr)

    pitched = librosa.effects.pitch_shift(y = y, sr= sr  , n_steps=np.random.randint(2,4), bins_per_octave=24)
    sf.write('./fastapi_app/test_aug/pitched.wav', pitched, sr)

    volumed = volume(y = y)
    sf.write('./fastapi_app/test_aug/volumed.wav', volumed, sr)