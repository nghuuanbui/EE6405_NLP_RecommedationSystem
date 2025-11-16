# Multi-Product Neural Retrieval Recommender System

A recommendation system supporting multiple product domains (Electronics and Beauty) with distinct neural architectures, unified through a modern web interface.

## Overview

This project implements a comprehensive recommendation pipeline from data preprocessing to model training and web deployment. The system leverages neural retrieval techniques to provide personalized product recommendations across different domains, demonstrating how architecturally diverse models can coexist within a single application framework.

**Key Features:**
- Dual-domain support (Electronics and Beauty products)
- Two distinct neural architectures (MLP Projector and Two-Tower)
- Interactive web interface with product browsing, search, and recommendations
- Modular codebase designed for scalability and extensibility
- Light and dark theme UI with responsive design

## Project Structure

```
.
├── data_preprocessing/
│   ├── data_preprocessing.py
├── model_training/
└── deployment/
    ├── app.py
    ├── requirements.txt
    ├── config/
    │   ├── __init__.py
    │   └── product_config.py
    ├── models/
    │   ├── __init__.py
    │   └── beauty_model.py
    ├── utils/
    │   ├── __init__.py
    │   └── beauty_loader.py
    ├── data/
    │   ├── Electronics/
    │   │   ├── items.parquet
    │   │   ├── mappings.json
    │   │   ├── item_text_emb_384.npy
    │   │   └── optionB_nlp_model/
    │   │       ├── item_projector.pt
    │   │       └── E_item_proj_128.npy
    │   └── Beauty/
    │       ├── items.parquet
    │       ├── mappings.json
    │       └── checkpoints/
    │           └── model_checkpoint.pt
    └── scripts/
        ├── verify_data.py
        └── generate_embeddings.py
```

## Data Preprocessing

The preprocessing pipeline prepares raw product metadata and review data for model training.

### Electronics Preprocessing
Transforms Amazon Electronics dataset into training-ready format with text embeddings and product mappings.

### Beauty Preprocessing
Processes Amazon Beauty dataset, creating item catalogs and preparing data for two-tower architecture training.

## Model Training

### Electronics Model
- **Architecture:** MLP-based projector
- **Input:** MiniLM embeddings (384-dim)
- **Output:** Projected embeddings (128-dim)
- **Training:** Contrastive learning with in-batch negatives

### Beauty Model
- **Architecture:** Two-tower (Item tower + User tower)
- **Item Tower:** RoBERTa-base encoder with learned projection
- **User Tower:** Transformer encoder with quality-aware features
- **Training:** InfoNCE loss with sentiment-derived quality signals

## Deployment

### Prerequisites

- Python 3.8+

### Installation

1. Navigate to the deployment directory:
```bash
cd deployment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Setup

Ensure the following data structure exists in `deployment/data/`:

**Electronics:**
- `Electronics/items.parquet` - Product catalog
- `Electronics/mappings.json` - ASIN to index mappings
- `Electronics/item_text_emb_384.npy` - MiniLM embeddings
- `Electronics/optionB_nlp_model/item_projector.pt` - Trained projector weights
- `Electronics/optionB_nlp_model/E_item_proj_128.npy` - Projected embeddings

**Beauty:**
- `Beauty/items.parquet` - Product catalog
- `Beauty/mappings.json` - ASIN to index mappings
- `Beauty/checkpoints/model_checkpoint.pt` - Trained two-tower model with embeddings

### Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Configuration

Product-specific settings can be modified in `config/product_config.py`:

```python
PRODUCTS = {
    'Electronics': {
        'name': 'Electronics',
        'icon': '⚡',
        'data_dir': './data/Electronics',
        'model_type': 'projector',
        'categories': { ... }
    },
    'Beauty': {
        'name': 'Beauty',
        'icon': '💄',
        'data_dir': './data/Beauty',
        'model_type': 'two_tower',
        'categories': { ... }
    }
}
```

## Usage

### Web Interface

1. **Product Selection:** Choose between Electronics and Beauty categories
2. **Browse Products:** Explore products by category with visual cards
3. **Search:** Find specific products using text search
4. **Manual Entry:** Input product ASINs directly
5. **Build Selection:** Add products to shopping cart
6. **Generate Recommendations:** Get personalized top-10 recommendations
7. **Export Results:** Download recommendations as CSV

## Architecture Details

### Frontend
- **Framework:** Streamlit
- **State Management:** Session state with cart persistence
- **Caching:** `@cache_resource` for efficient model loading
- **Theming:** Custom CSS with light/dark modes

### Backend
- **Models:** PyTorch-based neural networks
- **Search:** Sentence Transformers for semantic search
- **Data:** Pandas for catalog management
- **Embeddings:** NumPy arrays for efficient similarity computation

### Modular Design
- `config/`: Product configurations and UI settings
- `models/`: Model loading and inference logic
- `utils/`: Data loading and helper functions
- `data/`: Product catalogs and trained models

## Performance

- Model loading: ~2-5 seconds (cached after first load)
- Recommendation generation: <1 second for 10 recommendations
- Search: <500ms for text-based queries

## Dependencies

### Core Libraries
```
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pyarrow>=12.0.0
```

See `deployment/requirements.txt` for complete list.
