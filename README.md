# Polimeromics: Integrated Analysis of Biological and Structural Data

**Polimeromics** is a data analysis project that integrates biological and structural information from the BIOGRID and RCSB PDB databases. It employs advanced processing and modeling tools to generate interactive visualizations and predictive models. The project focuses on predicting the oligomeric state of proteins, a critical aspect in molecular biology and biomedical research.

---

## **Interactive Dashboards**

The project features two dashboards designed to explore data and provide detailed insights into the analyses:

1. **[Polimeromics Dashboard](https://polimeromic-production.up.railway.app/)**:
   - Displays general information about the processed data with interactive navigation.
   - Includes comparative graphs and details of molecular interactions from BIOGRID.
   - Explores structural and biophysical features derived from RCSB PDB.

2. **[Comparative Analysis Dashboard](https://comparativeanalysispolimeromic-production.up.railway.app/)**:
   - Integrates data from both databases, highlighting biological and structural relevance.
   - Provides detailed analyses of predictive patterns and critical features for the oligomeric state.
   - Features specific sections for dynamic visualization and molecular interaction analysis.

---

## **Analysis Workflow**

### 1. **BIOGRID Data**
- **Source**: [BIOGRID](https://thebiogrid.org/)
- **Original file**: `BIOGRID-ORCS-ALL1-homo_sapiens-1.1.16.screens` in `.tab.txt` format (~2.45 GB).
- Converted to `.csv`, resulting in ~10 GB, processed in six blocks due to computational constraints.
- **Cleaning techniques applied**:
  - Duplicate removal.
  - Missing value imputation using `data.fillna` and `data.median`.
  - Scaling and normalization of values.
  - Removal of irrelevant columns and special characters.

### 2. **RCSB PDB Data**
- **Source**: [RCSB PDB](https://www.rcsb.org/)
- Contains structural information of macromolecules (proteins and nucleic acids).
- Key features like pH, temperature, and solvent content were included.

### 3. **Data Fusion**
- Fusion process documented in: [Biogrid_fusion.ipynb](https://github.com/kentvalerach/Polimeromic/blob/main/Notebooks/Biogrid_fusion.ipynb).
- Validated for integrity and biological relevance.

---

## **XGBoost Predictive Model**

### **Objective**
- Predict the **oligomeric state** of proteins by integrating BIOGRID and RCSB data.
- This state is critical for understanding:
  - Protein biological functions.
  - Associations with diseases such as Alzheimer's and certain cancers.

### **Features**
1. **Data Integration**:
   - Combines molecular interaction data (BIOGRID) with biophysical features (RCSB).
   - Alignment and normalization of key identifiers such as protein and macromolecule names.
2. **Model Training**:
   - Chosen model: **XGBoost**, selected for its ability to handle heterogeneous data.
   - Evaluation metrics: Precision, recall, and F1-score.
3. **Expected Results**:
   - Predictive tool for identifying oligomeric states.
   - Applications in biomedical research and personalized medicine.

---

## **Project Documentation**

- **Notebooks**: Scripts for cleaning and analysis, organized by blocks:
  - [Notebooks](https://github.com/kentvalerach/Polimeromic/tree/main/Notebooks)
- **Reports**: Cleaning and verification details:
  - [Reports](https://github.com/kentvalerach/Polimeromic/tree/main/reports)
- **Visualizations**: Additional plots linked to each script.
  - [Reports](https://github.com/kentvalerach/Polimeromic/tree/main/Visualizations)

---

## **Deployment**

### **Requirements**
1. Python 3.11
2. Libraries specified in `requirements.txt`.

### **Basic Commands**
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python dashboard.py

# Production deployment (using Gunicorn)
gunicorn dashboard:server --workers=4 --bind=0.0.0.0:8000

