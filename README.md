# Fund Substitution Recommender System

This repository contains a complete system for recommending substitute investment funds based on their characteristics, performance metrics, and similarity within a learned embedding space. The solution combines machine learning clustering techniques with a user-friendly Streamlit web application to assist investors or analysts in identifying comparable funds when a particular one does not meet certain preferences.

## Project Overview

Investment funds often exhibit similar behavior or structure, but they may differ in features such as management fees, currency, or risk level. This application solves the problem of fund substitution by:

1. Preprocessing and filtering a master fund database and NAVs (Net Asset Values).
2. Computing financial metrics (e.g., cumulative return and volatility) for each fund.
3. Reducing dimensionality of the fund space using UMAP.
4. Clustering similar funds using DBSCAN and KMeans.
5. Allowing users to input an ISIN and receive a list of similar funds, filtered by criteria they wish to exclude (e.g., high fees, specific regions).
6. Providing a visual and tabular comparison of the original fund and its recommended substitutes.

## Techniques Used

* UMAP (Uniform Manifold Approximation and Projection) for non-linear dimensionality reduction while preserving local structure.
* DBSCAN for discovering dense clusters without requiring predefined cluster numbers.
* KMeans for reassigning small/noise clusters to the nearest well-defined centroids.
* Euclidean distance for recommending funds most similar to the target one within its cluster.
* Standardization and One-Hot Encoding for handling heterogeneous financial data (categorical + numerical).

## Project Structure

fund-substitution-recommender
├── utils/
│   └── preprocess\_dataset.py
├── app.py
├── clusterized\_funds.csv
├── requirements.txt
└── README.md

Note: Some data files (e.g., navs.pickle, maestro.csv, MSCI.csv) are not included due to licensing or privacy concerns.

## Required Files (Not Included)

To run the project locally, you must place the following files in the project directory:

* navs.pickle: Pickle file with historical NAVs for all funds. Columns = allfunds\_id; Index = date.
* maestro.csv: Master table with metadata and characteristics for each fund (e.g., asset type, region, ISIN).
* MSCI.csv: Supplementary file with index mappings (optional depending on your preprocessing function).

These files are not provided in the repository due to confidentiality. Ensure you maintain their original column structure as expected in the preprocess\_dataset function.

## Running the App

1. Install dependencies:

pip install -r requirements.txt

2. Launch the Streamlit app:

streamlit run app.py

3. Interface:

* In the sidebar, input the ISIN of the fund you want to replace.
* Review its characteristics.
* Choose two criteria you wish to exclude or improve (e.g., reduce fees or change region).
* The system will return a ranked list of similar funds within the same cluster, ordered by similarity.

## Example Criteria

You can choose to exclude funds based on:

* Asset Type
* Currency
* Geographical Zone
* Management Fee
* Ongoing Charges
* Minimum Return
* Maximum Volatility

The results will be filtered accordingly and sorted by similarity to the target fund based on Euclidean distance in the normalized space.

## Customization

You can customize:

* The number of neighbors used in UMAP.
* DBSCAN’s eps and min\_samples for clustering.
* The number of suggested substitutes (num\_sustitutivos in obtener\_fondos\_sustitutivos()).
* The list of numerical features used to compute similarity.
* The filtering criteria presented in the sidebar.

## Limitations

* The system depends on the quality and consistency of the NAVs and metadata.
* No model interpretability is provided beyond cluster assignment and distance.
* Assumes funds in the same cluster are comparable in nature (no validation of financial suitability).

## Author

Luis Marti Ávila

## License

This project is licensed for academic and demonstration purposes. Please contact the author for permission to use the methodology or interface commercially.
