from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def train_nn(X_train, y_train):
    nn = Sequential([
        Dense(32, input_dim=X_train.shape[1], activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    nn.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32,
           callbacks=[early_stop], verbose=1)
    return nn

def predict_nn(nn, X):
    return (nn.predict(X) > 0.5).astype(int)
