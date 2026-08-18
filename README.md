\# Retail Product Detection and Grouping Pipeline



A computer vision pipeline for detecting retail products from shelf images, extracting product text using OCR, grouping visually or textually similar products, and returning structured JSON results through a Flask API.



\## Overview



The application processes a retail shelf image through multiple computer vision stages:



1\. Product detection using YOLO

2\. Product cropping from detected bounding boxes

3\. OCR-based text extraction using EasyOCR

4\. Visual feature extraction using ResNet50

5\. Product grouping using OCR and visual similarity

6\. Group ID assignment

7\. Annotated image generation

8\. JSON response generation



\## Pipeline



```text

Input Image

&#x20;    │

&#x20;    ▼

YOLO Product Detection

&#x20;    │

&#x20;    ▼

Bounding Box Extraction

&#x20;    │

&#x20;    ├──────────────► Product Crops

&#x20;    │

&#x20;    ▼

EasyOCR Text Extraction

&#x20;    │

&#x20;    ▼

ResNet50 Feature Embeddings

&#x20;    │

&#x20;    ▼

Product Similarity \& Grouping

&#x20;    │

&#x20;    ▼

Group ID Assignment

&#x20;    │

&#x20;    ├──────────────► Annotated Image

&#x20;    │

&#x20;    └──────────────► JSON Response

```



\## Technologies Used



\* Python

\* Flask

\* Ultralytics YOLO

\* EasyOCR

\* PyTorch

\* Torchvision

\* ResNet50

\* NumPy

\* Pillow



\## Project Structure



```text

.

├── app.py

├── requirements.txt

├── README.md

├── .gitignore

│

├── models/

│   └── detector/

│       ├── candidate\_v2/

│       │   └── candidate\_v2.pt

│       └── detector utility scripts

│

├── templates/

│   └── index.html

│

└── uploads/

&#x20;   └── .gitkeep

```



\## Installation



Create and activate a Python virtual environment:



```bash

python -m venv .venv

```



Activate it on Windows:



```bash

.venv\\Scripts\\activate

```



Install the required dependencies:



```bash

pip install -r requirements.txt

```



\## Running the Application



Start the Flask application:



```bash

python app.py

```



The application runs locally at:



```text

http://127.0.0.1:5000

```



Open the address in a browser to access the application interface.



\## API



\### `GET /`



Loads the web interface.



\### `POST /detect`



Accepts an image and performs the complete detection and grouping pipeline.



The response contains:



\* Total detected objects

\* Total product groups

\* Object IDs

\* Bounding-box coordinates

\* Group IDs

\* Path to the annotated image



\## Grouping Logic



The grouping process uses two complementary signals:



\### 1. OCR-based grouping



Objects with matching cleaned OCR text are assigned to the same group.



\### 2. Visual similarity fallback



ResNet50 embeddings are generated for detected product crops. Cosine-style similarity is calculated using normalized embeddings, and the visual similarity threshold is used when OCR information is unavailable or insufficient.



The current visual similarity threshold is:



```text

0.78

```



The detection confidence threshold is:



```text

0.50

```



\## Model and Data



The detection model is stored under:



```text

models/detector/candidate\_v2/candidate\_v2.pt

```



The application uses CPU inference for both YOLO and the ResNet50 embedding model.



The sample images supplied as part of the assignment are intentionally not included in this repository.



\## Output



For each processed image, the application generates:



\* Detected product bounding boxes

\* Group identifiers

\* Annotated output image

\* Structured JSON response



\## Notes



This repository contains the implementation and supporting files required for the assignment submission.



Assignment-provided sample data is not included in the repository.



