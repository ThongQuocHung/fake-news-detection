# Fake News Detection using Graph Neural Networks

## Overview
This project implements fake news detection using **Graph Neural Networks (GNNs)** on the **UPFD dataset (Twitter-Based Fake News Dataset)**. The dataset consists of news articles propagated on social media platforms, classified as real or fake. The models are trained and evaluated on **Politifact** and **Gossipcop** datasets.

## Features
- Uses **Graph Attention Networks (GAT)**, **Graph Convolutional Networks (GCN)**, and **GraphSAGE** for fake news detection.
- Preprocessing and loading of the **UPFD dataset**.
- Model training and evaluation with **accuracy, F1-score, and AUC-ROC**.
- Implementation in **PyTorch Geometric**.

## Installation
Make sure you have the required dependencies installed:
```sh
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.1+cu116.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.1+cu116.html
pip install torch-geometric
pip install -q git+https://github.com/snap-stanford/deepsnap.git
```

## Dataset
The dataset is loaded directly from UPFD and consists of two sources:
1. **Politifact** - News fact-checked by PolitiFact.
2. **Gossipcop** - News fact-checked by GossipCop.

## Running the Project
1. Clone the repository:
```sh
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```
2. Run the Jupyter notebooks:
   - `Fake_News_Detection_Politifact.ipynb`
   - `Fake_News_Detection_Gossipcop.ipynb`
3. Train the model and evaluate its performance.

## Model Architecture
The models are based on **Graph Neural Networks (GNNs)** using:
- **Graph Attention Networks (GAT)**
- **Graph Convolutional Networks (GCN)**
- **GraphSAGE**
- **Dense SAGEConv with DiffPool**

## Evaluation Metrics
The models are evaluated using:
- **Accuracy**
- **F1-Score**
- **ROC-AUC Score**

## Results
The results of the models will be displayed in terms of classification performance on both datasets.

## Contributors
- [Quoc Hung]

## License
This project is licensed under the MIT License.

