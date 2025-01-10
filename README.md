# Automated Property Valuation Model for Denmark

## Project Description
This project aims to develop an Automated Valuation Model (AVM) to estimate the sales prices of owner-occupied apartments in Denmark. By leveraging a dataset of over 100,000 property transactions (January 2015 - June 2024), we will create a robust, data-driven model to predict property prices based on key features such as location, size, amenities, and proximity to natural features.

## Overall Goal
The primary objective is to design a model that provides accurate and reliable property valuations. The model will enhance property market analysis by incorporating advanced data science techniques to analyze a comprehensive dataset of property transactions.

## Framework
- **LightGBM**: Selected for its efficiency with large tabular datasets and strong interpretability in feature importance analysis.
- **PyTorch Tabular**: Integrated as a third-party PyTorch-based framework to explore deep learning techniques for tabular data, enabling a comparison of traditional and neural network-based methods.

## Data
The dataset includes detailed property transaction records, encompassing:
- **Attributes**: Address, zip code, municipality, building area, number of rooms, distance to harbors, lakes, and coastlines, and amenities such as elevators and bathrooms.
- **Feature Engineering**: 
  - New variables like price per square meter and proximity to city centers.
  - Interaction terms between features to capture complex relationships.

## Models
We will explore two modeling approaches:
1. **Gradient Boosting with LightGBM**:
   - Effective for tabular data modeling.
   - Provides interpretability and computational efficiency.
2. **Neural Networks with PyTorch Tabular**:
   - Captures non-linear relationships and complex feature interactions.
   - Leverages flexibility in combining traditional and neural approaches.

## Deliverables
- **Trained AVM**:
  - Predictions on a test dataset.
- **Model Deployment**:
  - REST API implementation using FastAPI for accessibility.
- **Comprehensive Documentation**:
  - Methodology and evaluation metrics to ensure transparency and reproducibility.
