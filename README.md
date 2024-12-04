# FaceRecognition

# FaceRecognition: A Face Recognition and Visualization Tool

## Project Overview

**FaceRecognition** is a Python-based project developed for CECS 406 Assignment 5. This application uses advanced tools for face detection, feature extraction, and dimensionality reduction to create a t-SNE map of face embeddings. It operates on a subset of the [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset, showcasing facial recognition and clustering in a 2D visualization.

---

## Table of Contents
1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Setup Instructions](#setup-instructions)
4. [Workflow](#workflow)
5. [Usage](#usage)
6. [Output](#output)
7. [References](#references)

---

## Features

- **Face Detection**: Detect faces using the [MTCNN](https://github.com/ipazc/mtcnn) (Multi-task Cascaded Convolutional Networks) model.
- **Feature Extraction**: Extract 2048-dimensional embeddings using the pre-trained [VGGFace2](https://github.com/rcmalli/keras-vggface) model.
- **Dimensionality Reduction**: Reduce high-dimensional embeddings to a 2D space using t-SNE from Scikit-learn.
- **Visualization**: Create a scatter plot grouping and coloring points by individual identities for easy interpretation.

---

## Prerequisites

Before running the project, ensure you have the following:

- Python 3.x
- Libraries:
  - `tensorflow`
  - `keras-vggface`
  - `mtcnn`
  - `scikit-learn`
  - `matplotlib`
  - `numpy`
  - `pandas`
- [LFW Dataset](https://scikit-learn.org/stable/datasets/real_world.html#labeled-faces-in-the-wild-dataset)
- Install the dependencies using the provided `environment.yml` file:
  ```bash
  conda env create -f environment.yml
  conda activate face_recognition
  ```

---

## Setup Instructions

1. **Clone or Download the Repository**:
   ```bash
   git clone https://github.com/your-repo/FaceRecognition.git
   cd FaceRecognition
   ```

2. **Install Dependencies**:
   Install required libraries as outlined above.

3. **Dataset Preparation**:
   The LFW dataset is automatically fetched using Scikit-learn's `fetch_lfw_people()` function.

---

## Workflow

### 1. **Data Loading and Preprocessing**
- **Load Dataset**: Use `fetch_lfw_people` to load the LFW dataset.
- **Face Detection**: Detect faces with MTCNN.
- **Preprocessing**:
  - Crop detected faces.
  - Resize images to 224×224 pixels.
  - Normalize pixel values to `[0, 1]`.

### 2. **Model Loading**
- **MTCNN Model**: Load the pre-trained MTCNN model for face detection.
- **VGGFace2 Model**: Load the VGGFace2 model for feature extraction using `keras-vggface`.

### 3. **Embedding Extraction**
- Extract 2048-dimensional embeddings for each detected face using VGGFace2.

### 4. **t-SNE Visualization**
- Reduce dimensionality of embeddings using t-SNE from 2048 to 2 dimensions.
- Generate a scatter plot where:
  - Each point represents a face embedding.
  - Points belonging to the same individual are grouped with distinct colors/markers.

### 5. **Display Results**
- Display the t-SNE map with labeled groups.

---

## Usage

1. **Run the Jupyter Notebook**:
   - Open the `.ipynb` file in Jupyter Notebook.
   - Execute the cells in sequence.

2. **Expected Outputs**:
   - t-SNE scatter plot showing embeddings grouped by individuals.

---

## Output

The final output of this project is a **2D scatter plot** generated using `matplotlib`. Key features include:
- **Grouping by Individual**: Each point represents a face embedding, with points from the same individual visually grouped.
- **Distinct Visuals**: Different colors or markers are used for clarity.
- **Legends/Labels**: Clearly indicate individual identities in the plot.

---

## References

1. **MTCNN**:
   - [GitHub Repository](https://github.com/ipazc/mtcnn)
   - [Tutorial](https://towardsdatascience.com/face-detection-with-mtcnn-5a9a56fca869)

2. **VGGFace2**:
   - [GitHub Repository](https://github.com/rcmalli/keras-vggface)
   - [Paper and Dataset](https://arxiv.org/abs/1710.08092)

3. **t-SNE**:
   - [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
   - [Understanding t-SNE](https://distill.pub/2016/misread-tsne/)

4. **LFW Dataset**:
   - [Dataset Page](http://vis-www.cs.umass.edu/lfw/)
   - [Usage in Python](https://scikit-learn.org/stable/datasets/real_world.html#labeled-faces-in-the-wild-dataset)

---

## Notes

- Ensure a GPU-enabled environment for faster processing, especially during face detection and embedding extraction.
- t-SNE is a computationally intensive process; performance might vary based on the system configuration. Adjust `perplexity` and other hyperparameters if needed for improved results.
