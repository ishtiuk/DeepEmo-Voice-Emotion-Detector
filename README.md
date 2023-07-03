# DeepEmo-Voice-Emotion-Detector
It's a deep learning LSTM model which can detect emotion from user's given audio clip.

## Emotion Detection from Audio Clips

I have developed a Deep Learning AI model (LSTM) capable of detecting emotions from audio clips. The model recognizes the following emotions: 'happy', 'sad', 'neutral', 'surprise', 'fear', 'angry', and 'disgust', resulting in a total of 7 categories.

## Requirements

Make sure you have the following dependencies installed:

- Python 3.8 or later
- NumPy
- librosa
- TensorFlow
- Keras
- Matplotlib
- Seaborn

You can install these dependencies by running the following command:

```bash
pip install numpy librosa tensorflow keras matplotlib seaborn
```

## Feature Extraction

To extract features from the audio files, I employed the following Python function:

```python
def feature_extract(data):
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

data, sr = librosa.load(path, duration=3, offset=0.5)
features = feature_extract(data)
```

## LSTM Model Architecture

The emotion detection model was trained using the LSTM architecture implemented with Keras and TensorFlow backend:

```python
model = Sequential()
model.add(LSTM(128, return_sequences=False, input_shape=(40, 1)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

model.summary()
```
This model achieved an impressive 99% accuracy at epoch 34.


## Prediction

You can use the following function to make predictions on your own audio clips:

```python
def predict_single_audio(path):
    emotion_dict = {0: 'happy', 1: 'sad', 2: 'neutral', 3: 'surprise', 4: 'fear', 5: 'angry', 6: 'disgust'}
    data, sr = librosa.load(path, duration=3, offset=0.5)
    x = feature_extract(data).reshape((1, 40, 1))

    return emotion_dict[np.argmax(model.predict(x))]
```

Feel free to utilize this function to predict the emotion from your audio clips!

## License

This project is licensed under the [MIT License](https://github.com/ishtiuk/DeepEmo-Voice-Emotion-Detector/blob/main/LICENSE).

