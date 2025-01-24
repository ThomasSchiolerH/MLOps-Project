# Automated Property Valuation Model for Denmark

## Project Description
This project aims to develop an Automated Valuation Model (AVM) to estimate the sales prices of owner-occupied apartments in Denmark. By leveraging a dataset of over 100,000 property transactions (January 2015 - June 2024), the goal is to create a robust and scalable machine learning pipeline to predict property prices based on features such as location, size, amenities, and proximity to natural features.

## Objectives
The primary objective of the project is to deliver an accurate and reliable property valuation system that incorporates MLOps best practices for scalability, reproducibility, and usability.

## Frameworks and Tools
To build and deploy the AVM, the project utilizes the following frameworks:
- **LightGBM**: Used as the primary machine learning framework for its efficiency with tabular data and feature interpretability.
- **scikit-learn**: Used for preprocessing tasks like scaling, data splitting, and evaluation metrics.
- **FastAPI**: Employed to create a REST API for serving predictions.
- **Gradio**: Integrated to provide a user-friendly web-based frontend for non-technical stakeholders to interact with the model.
- **Google Cloud Platform (GCP)**: Used for deploying the application with services like Cloud Run, Vertex AI, and Cloud Storage.
- **DVC (Data Version Control)**: Implemented for versioning datasets and ensuring reproducibility.

## Data
The dataset comprises over 100,000 property transactions, including features such as:
- **Location Information**: Address, zip code, and municipality.
- **Property Features**: Size, number of rooms, construction and rebuilding years, and the presence of amenities like elevators.
- **Geographical Features**: Proximity to lakes, harbors, and coastlines.
- **Target Variable**: Price per square meter (SQM_PRICE).

Initial preprocessing includes cleaning the data, handling missing values, and feature engineering, such as extracting date-related features and creating interaction terms.

## Models
The project focuses on the following models:
1. **LightGBM Regression**: Selected for its speed, ability to handle categorical data, and interpretability.
2. **PyTorch-based Models**: Explored for potential use in capturing non-linear relationships and comparing performance with traditional machine learning models.

The LightGBM model serves as the primary choice due to its efficiency and ability to handle large datasets effectively.

## Deliverables
- A trained LightGBM model for property price prediction.
- A REST API developed using FastAPI for serving predictions.
- A Gradio-based web interface for easy interaction with the model.
- Deployment of the model and API on Google Cloud Run, ensuring scalability and accessibility.
- Comprehensive documentation detailing the methodology, results, and instructions for reproducing the project.

This project delivers a robust, end-to-end solution for automated property valuation, combining machine learning, cloud deployment, and MLOps practices.
