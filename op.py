import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models


def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (224, 224))  
            images.append(img)
            
            label = 1 if 'cancer' in filename else 0
            labels.append(label)
    return np.array(images), np.array(labels)

folder_path = r"C:\Users\DELL\OneDrive\Pictures\Screenshots"  
images, labels = load_images_from_folder(folder_path)


genomic_data = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\genomic_data.csv")
X_genomic = genomic_data.drop('target', axis=1).values
y_genomic = genomic_data['target'].values


X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(images, labels, test_size=0.2)
X_train_genomic, X_test_genomic, y_train_genomic, y_test_genomic = train_test_split(X_genomic, y_genomic, test_size=0.2)


def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid') 
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


cnn_model = create_cnn_model()
cnn_model.fit(X_train_img, y_train_img, epochs=10, batch_size=32, validation_split=0.2)


rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_genomic, y_train_genomic)

y_pred_genomic = rf_model.predict(X_test_genomic)
print(classification_report(y_test_genomic, y_pred_genomic))

def predict_risk(image, genomic_features):
   
    image = cv2.resize(image, (224, 224)) / 255.0
    image = np.expand_dims(image, axis=0)  

    
    img_prediction = cnn_model.predict(image)[0][0]
    genomic_prediction = rf_model.predict([genomic_features])[0]

    combined_prediction = (img_prediction + genomic_prediction) / 2
    return combined_prediction

def alert_provider(risk_score, threshold=0.5):
    if risk_score > threshold:
        print("Alert: High risk of cancer detected! Recommend further screening.")
    else:
        print("Risk is low. No immediate action required.")


sample_image_path = r"C:\Users\DELL\Downloads\cancer_2.jpg"  
sample_image = cv2.imread(sample_image_path)
sample_genomic = X_test_genomic[0]  

risk = predict_risk(sample_image, sample_genomic)
print(f'Combined cancer risk score: {risk}')
alert_provider(risk)
