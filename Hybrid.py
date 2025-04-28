import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf

# Veri setini yükle
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Veriyi CNN için hazırla
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Etiketleri kategorik forma dönüştür (CNN eğitimine hazırlık için)
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# Özellik çıkarımı için CNN modeli oluştur
feature_extractor = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten()
])

# Tam CNN modeli (sadece karşılaştırma için)
full_cnn = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# CNN modelini derle ve eğit
full_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
full_cnn.summary()

# Tüm veriyi CNN ile eğitmek yerine, sadece bir kısmını kullanabiliriz (hız için)
sample_size = 10000
x_train_sample = x_train[:sample_size]
y_train_cat_sample = y_train_cat[:sample_size]

full_cnn.fit(x_train_sample, y_train_cat_sample, epochs=3, batch_size=128, validation_split=0.2)

# Özellik çıkarımı (CNN'in son katmanını kullanarak)
X_train_features = feature_extractor.predict(x_train)
X_test_features = feature_extractor.predict(x_test)

# Oluşturulan özellikleri ve etiketleri .npy dosyalarına kaydet
np.save('X_train_features.npy', X_train_features)
np.save('y_train.npy', y_train)
np.save('X_test_features.npy', X_test_features)
np.save('y_test.npy', y_test)

print(f"Özellik boyutu: {X_train_features.shape[1]}")

# Kaydedilen özellikleri ve etiketleri yükle
X_train_features = np.load('X_train_features.npy')
y_train = np.load('y_train.npy')
X_test_features = np.load('X_test_features.npy')
y_test = np.load('y_test.npy')

print("SVM modeli eğitiliyor...")
clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')

# Daha hızlı eğitmek amacıyla örneklerin bir kısmı
sample_size = 10000  # Örnek boyutu
X_train_sample = X_train_features[:sample_size]
y_train_sample = y_train[:sample_size]

clf.fit(X_train_sample, y_train_sample)

# Tahmin
y_pred = clf.predict(X_test_features)

# Sonuçları değerlendirme
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"SVM doğruluk oranı: {accuracy:.4f}")

# Sınıflandırma raporu
report = metrics.classification_report(y_test, y_pred)
print("\nSınıflandırma Raporu:")
print(report)

# Karmaşıklık matrisi
cm = metrics.confusion_matrix(y_test, y_pred)
