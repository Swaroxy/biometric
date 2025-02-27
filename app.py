import os
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import zipfile
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# Paths for dataset and models
dataset_path = "/home/ubuntu/biometric/"
zip_path = os.path.join(dataset_path, "Tr0.zip")
images_folder = "Tr0"
npy_data_path = dataset_path
data_path = os.path.join(npy_data_path, "yaleExtB_data.npy")
target_path = os.path.join(npy_data_path, "yaleExtB_target.npy")
single_image_path = "D:/biometric/Tr0/yaleB02_P00A+000E+00.jpg"  # Path to the single image

# Function to process images and save as .npy files
def process_images():
    if not os.path.exists(os.path.join(dataset_path, images_folder)):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        print(f"Extracted '{zip_path}' to '{dataset_path}'")
    else:
        print(f"'{images_folder}' folder already exists. Skipping extraction.")

    images_folder_path = os.path.join(dataset_path, images_folder)
    fls = os.listdir(images_folder_path)
    n = len(fls)
    print(f'Number of images: {n}')

    im1 = image.imread(os.path.join(images_folder_path, fls[0]))
    print(im1.shape)
    plt.imshow(im1, cmap=plt.cm.gray)
    plt.show()

    m = im1.shape[0] * im1.shape[1]
    images_data = np.zeros((n, m))
    images_target = np.zeros((n,))

    for i in range(n):
        filename = fls[i]
        img = image.imread(os.path.join(images_folder_path, filename))
        images_data[i] = np.ravel(img)
        c = int(filename[5:7])
        images_target[i] = c

        if i % 100 == 0:
            print(f"Processed {i}/{n} images")

    np.save(data_path, images_data)
    np.save(target_path, images_target)
    print(f"Data saved to: {data_path}")
    print(f"Target labels saved to: {target_path}")
    print("Images data and target labels:")
    print(f"Data shape: {images_data.shape}")
    print(f"Target shape: {images_target.shape}")
    print("Files in the dataset directory:", os.listdir(dataset_path))

# Function to train and evaluate the MLPClassifier
def train_and_evaluate():
    print("Loading data...")
    data = np.load(data_path)
    target = np.load(target_path)
    print(f"Data shape: {data.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Number of unique classes: {len(np.unique(target))}")

    X, y = data, target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    nof_prin_components = 150
    print("Performing PCA...")
    pca = PCA(n_components=nof_prin_components, whiten=True).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    nohn = 300
    print("Training the classifier...")
    clf = MLPClassifier(hidden_layer_sizes=(nohn,), solver='adam', activation='relu', 
                        batch_size=256, verbose=True, early_stopping=True, random_state=42)
    clf.fit(X_train_pca, y_train)

    print("Evaluating the classifier...")
    y_pred = clf.predict(X_test_pca)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Performing cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train_pca, y_train, cv=kf, scoring='accuracy')
    print(f"Cross-validation scores: {scores}")
    print(f"Mean cross-validation accuracy: {np.mean(scores)}")

    return clf, pca  # Return the trained classifier and PCA model

# Function to predict the class of a single image
def predict_single_image(clf, pca, image_path):
    img = image.imread(image_path)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

    # Flatten the image
    image_data = np.ravel(img).reshape(1, -1)
    print(f"Processed image shape: {image_data.shape}")

    # Transform the image using the PCA model
    image_pca = pca.transform(image_data)

    # Predict the class using the trained classifier
    prediction = clf.predict(image_pca)
    print(f"Predicted class: {prediction[0]}")
    return prediction[0]

if __name__ == '__main__':
    process_images()
    clf, pca = train_and_evaluate()
    predict_single_image(clf, pca, single_image_path)