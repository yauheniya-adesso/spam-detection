import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# 1️⃣ Load dataset
df = pd.read_csv('sms+spam+collection/SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

emoji_map = {
    'ham': '✅ ham',
    'spam': '❗spam'
}

# 2️⃣ Encode labels (ham=0, spam=1)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# 3️⃣ Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# 4️⃣ Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Convert sparse matrix to dense
X_train_tfidf = X_train_tfidf.toarray()
X_test_tfidf = X_test_tfidf.toarray()

# 5️⃣ Build neural network using functional API
input_layer = tf.keras.Input(shape=(X_train_tfidf.shape[1],), name='Input')
hidden = tf.keras.layers.Dense(512, activation='relu', name='Hidden')(input_layer)
hidden = tf.keras.layers.Dropout(0.5, name='Dropout')(hidden)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='Output')(hidden)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# 6️⃣ Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 7️⃣ Train model
history = model.fit(
    X_train_tfidf,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test_tfidf, y_test)
)

model.save('model_weights.keras')

# 8️⃣ Evaluate model
y_pred = (model.predict(X_test_tfidf) > 0.5).astype("int32")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Show 4 random ham and 4 random spam messages
np.random.seed(2025)  # for reproducibility

# Get indices for ham and spam
ham_indices = np.where(y_test == 0)[0]
spam_indices = np.where(y_test == 1)[0]

# Randomly pick 4 from each
random_ham = np.random.choice(ham_indices, size=4, replace=False)
random_spam = np.random.choice(spam_indices, size=4, replace=False)
random_indices = np.concatenate([random_ham, random_spam])
np.random.shuffle(random_indices)  # shuffle to mix ham and spam

print("\nSample Predictions:\n")
for idx in random_indices:
    message = X_test.iloc[idx]
    true_label = label_encoder.inverse_transform([y_test.iloc[idx]])[0]
    predicted_label = label_encoder.inverse_transform([y_pred[idx][0]])[0]
    true_label_emoji = emoji_map.get(true_label, true_label)
    predicted_label_emoji = emoji_map.get(predicted_label, predicted_label)
    print(f"Message: {message}")
    print(f"True label: {true_label_emoji}, Predicted label: {predicted_label_emoji}")
    print("-" * 80)


# 9️⃣ Plot accuracy over epochs
import matplotlib.pyplot as plt
plt.style.use("dark_background")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])
plt.savefig('learning_curve.png', dpi=300, bbox_inches="tight")
