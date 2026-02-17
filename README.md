# Human Face Emotion Recognition using Transfer Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Transfer%20Learning-red)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue)


![image alt](https://github.com/aryanvaghsiya11-a11y/Emotion-Detection-Transfer-Learning/blob/d56e71606387c7994370f2ca05990840a694d250/Emotion-Detection.jpg)

## 📌 Project Overview

This project implements a Deep Learning pipeline to recognize and classify human facial emotions. It leverages **Transfer Learning** techniques to achieve high accuracy by utilizing pre-trained models on the **Human Face Emotions** dataset.

The workflow includes automated data acquisition from Kaggle, data preprocessing, visualization, and model training to identify various emotional states (e.g., Happy, Sad, Angry) from facial images.

## 📂 Dataset

The project uses the [Human Face Emotions Dataset](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions) from Kaggle.

- **Structure**: The dataset is organized into subdirectories where each folder name corresponds to an emotion class.
- **Input**: RGB images of human faces.
- **Classes**: Includes standard emotions such as *Anger, Contempt, Disgust, Fear, Happiness, Sadness, Surprise*.

## 🛠️ Technologies Used

- **Language**: Python 3
- **Environment**: Jupyter Notebook / Google Colab
- **Libraries**:
  - `opencv-python`: For image reading and processing.
  - `matplotlib`: For data visualization and plotting.
  - `kaggle`: For programmatic dataset downloading.
  - `zipfile` & `os`: For file handling and directory management.
  - *(Implied)* `tensorflow` / `keras` or `pytorch`: For building the Transfer Learning model.

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/Human-Face-Emotion-Recognition.git](https://github.com/your-username/Human-Face-Emotion-Recognition.git)
cd Human-Face-Emotion-Recognition
```
## Data Acquisition (Kaggle API)

The notebook begins by setting up the environment to download the dataset programmatically.

Setup: It creates the hidden directory ~/.kaggle and copies the kaggle.json **API** token there to authenticate.

Download: It executes the command !kaggle datasets download samithsachidanandan/human-face-emotions.

Source: The output confirms the dataset was successfully downloaded from samithsachidanandan/human-face-emotions.

## Data Preparation

Extraction: The script uses Python's built-in zipfile library to unzip human-face-emotions.zip into the /content directory.

Directory Structure: It identifies that the extracted data resides in /content/Data.

## Exploratory Data Analysis (Visualization)

The core code block in this file is a visualization function designed to verify the dataset integrity.

Libraries: It imports os (for file navigation), random (for sampling), cv2 (OpenCV for image loading), and matplotlib.pyplot (for display).

Function show_images:

It takes a class name (emotion label) as input.

It lists all images in that class's folder.

It randomly selects 5 images.

It converts the images from **BGR** (OpenCV format) to **RGB** (Matplotlib format).

It displays them in a horizontal grid.

Execution: The notebook iterates through the folders in /content/Data and calls this function, successfully printing *Displaying Random Samples per Class :* followed by the image grids .

Summary of Missing Components To achieve accuracy metrics (e.g., *95% Accuracy*), the following code blocks need to be added to this notebook:

Data Generators: ImageDataGenerator to rescale and augment the images.

Model Definition: Loading a base model like **VGG16** or ResNet50 (Transfer Learning).

Training: compiling the model with an optimizer (e.g., Adam) and loss function (e.g., Categorical Crossentropy), then running model.fit().

Evaluation: Running model.evaluate() to print the final accuracy and loss.
