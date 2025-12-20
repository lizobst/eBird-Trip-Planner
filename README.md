
# Texas Bird Detection System

*Predictive modeling of bird detection probabilities across South Texas using machine learning and citizen science data*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ebird-trip-planner.streamlit.app/)

<img width="1910" height="993" alt="image" src="https://github.com/user-attachments/assets/8ce3d768-cf64-4191-a0d5-e70197a030c1" />


---

## Project Overview

This project develops a scalable machine learning system to predict bird detection probabilities across Bexar and Hidalgo counties in Texas, enabling data-driven trip planning for birders. Using over 200,000 citizen science observations from eBird (2022-2025), the system trains 281 seasonal models achieving an average ROC-AUC of 0.842.

**Key Innovation:** A species classification framework based on spatial variance separates habitat specialists (where location matters) from ubiquitous species (where it doesn't), focusing ML efforts on 90 species where spatial modeling adds value.

---

## Problem Statement

Us birders face a common challenge: *Where and when should I go to maximize my chances of seeing specific bird species?*

Traditional approaches rely on:
- Anecdotal reports from other birders
- General range maps (too broad)
- Recent sightings (limited context)

**This system provides:** Probabilistic predictions integrating spatial habitat features, temporal patterns, and real-time observations.


*The initial motivation for this project is to maximize the number of bird species I will see in 2026.*

---

## Methodology

### Data Sources
- **eBird Complete Checklists:** 200,000+ observations across 500+ species (2022-2025)
- **Geospatial Data:** 
  - NLCD land cover (30m resolution)
  - NHD water bodies and watersheds
  - Protected areas (parks, refuges)
- **Study Areas:** 
  - Bexar County (urban/hill country, 900 sq mi)
  - Hidalgo County (Rio Grande Valley subtropical, 1,600 sq mi)

### Feature Engineering

**Spatial Features (78% avg importance):**
- Latitude, longitude, grid cell location
- Distance to nearest water body
- Land cover type (urban, forest, grassland, wetland, agricultural)
- Protected area indicator

**Temporal Features:**
- Season (winter, spring, summer, fall)
- Hour of day
- Week of year (for seasonal models)

**Critical Decision:** Removed effort-based features (duration, distance, observers) to avoid learning observer behavior instead of bird ecology.

### Species Classification System

Not all species benefit from spatial modeling. A five-tier classification system based on **spatial coefficient of variation (CV)** determines modeling strategy:

| Tier | Criteria | Strategy | Count | Example Species |
|------|----------|----------|-------|-----------------|
| **1A: Spatial Specialists** | 10-40% detection, spatial CV >0.4 | Seasonal models | 18 | Green Kingfisher |
| **1B: Seasonal Specialists** | <10% overall but >10% in season | Seasonal models | 32 | Painted Bunting |
| **2A: Temporal Only** | Uniform distribution, spatial CV <0.3 | Statistics only | 63 | Couch's Kingbird |
| **2B: Too Common** | >40% detection everywhere | Statistics only | 12 | Northern Cardinal |
| **3: Too Rare** | <2% detection | Skip | 341 | Vagrant species |
| **4: County Specific** | Present in one county only | County models | 10 | Green Jay (Hidalgo) |
| **5: Cross-County** | Common in both counties | Combined models | 30 | Red-tailed Hawk |

**Result:** 90 modelable species where spatial predictions outperform simple averages.

### Model Architecture

**Algorithm:** Random Forest Classifier
- 100 trees
- Max depth: 10
- Min samples per leaf: 20
- Class weight: balanced (handles detection rate imbalance)

**Calibration:** Platt scaling ensures predicted probabilities reflect true detection rates

**Seasonal Strategy:** 1-4 models per species depending on presence patterns
- Year-round residents: 4 seasonal models
- Migratory species: 1-2 seasonal models (only present seasons)

**Validation:** 80/20 train-test split, minimum ROC-AUC threshold of 0.70

---

## Results

### Model Performance

**Overall Statistics:**
- **281 models trained** across 90 species
- **Average ROC-AUC: 0.842** (range: 0.715-0.964)
- **Spatial feature importance: 78%** (validates location-based approach)
- **Min ROC-AUC: 0.715** (even weakest models exceed threshold)

**Top Performers (ROC-AUC >0.95):**
- Wild Turkey: 0.964
- Yellow-crowned Night Heron: 0.963
- Black-necked Stilt: 0.952
- Northern Shoveler: 0.943

**Distribution by County:**
- Bexar: 21 models (avg 0.824)
- Hidalgo: 71 models (avg 0.837)
- Combined: 158 models (avg 0.840)

### Feature Importance Analysis

**Spatial features dominate predictions:**
- Latitude + Longitude + Grid Cell: 40-60%
- Distance to water: 10-25% (especially for waterfowl)
- Land cover: 10-20%
- Protected area: 3-8%

**Case Study - Black Phoebe (Hidalgo, Fall):**
- Distance to water: **25.3%** ← Strong water dependence
- Longitude: 23.9%
- Latitude: 22.4%
- Total spatial: **79.5%**
- ROC-AUC: 0.797

This validates the ecological relationship: Black Phoebes are strongly associated with water.

### Cross-County Insights

**County Specialties Successfully Identified:**
- **Green Jay:** 58% detection in Hidalgo, 1% in Bexar (Rio Grande specialty)
- **Golden-cheeked Warbler:** 8% in Bexar, 0% in Hidalgo (hill country endemic)
- **Green Kingfisher:** 18% in Hidalgo, 2% in Bexar (subtropical preference)

**Shared Species Show Habitat Differences:**
- Red-tailed Hawk: Modeled in both counties with county indicators
- Model learns county-specific patterns automatically

### Seasonal Modeling Impact

**Rescued "rare" species through seasonal focus:**
- **Painted Bunting:** 4% overall → 28% in summer (breeding season)
- **Ruby-crowned Kinglet:** 3% overall → 15% in winter (migratory)
- **Broad-winged Hawk:** 7% overall → 14% in spring (migration)

Without seasonal models, these 32 species would have been excluded from analysis.

---

## Application: Trip Planning Tool

### Interactive Web Application

**Deployed on Streamlit Cloud:** [https://ebird-trip-planner.streamlit.app/]

**User Workflow:**
1. Select county (Bexar or Hidalgo)
2. Choose trip date (determines season)
3. Select time of day (5 AM - 7 PM)
4. View ranked species list with detection probabilities

**Outputs:**
- **Top 20 species** most likely to be seen
- **Probability tiers:** Excellent (>30%), Very Good (20-30%), Good (10-20%)
- **County specialties** highlighted
- **Model quality indicators** (ROC-AUC scores)

### Real-Time Data Integration

**eBird API Integration:**
- Recent notable sightings (7-day window)
- County-wide rare bird alerts
- Hotspot-specific activity
- Validation: Predictions vs actual observations

**Featured Hotspots:**
- **Bexar:** Mitchell Lake, Friedrich Wilderness Park, OP Schnabel Park
- **Hidalgo:** Estero Llano Grande, Bentsen State Park, Edinburg Wetlands

---

## Key Findings

### 1. Location Matters for Habitat Specialists

Species with strong habitat associations show high spatial feature importance:
- **Water birds:** Distance to water = 20-30% importance
- **Forest species:** Land cover forest = 15-25%
- **Wetland species:** Wetland cover + water proximity = 40%+

### 2. Spatial Variance Predicts Modelability

**Species with high spatial CV (>0.5):** Excellent models (ROC-AUC >0.85)
- Wild Turkey, Green Kingfisher, Yellow-crowned Night Heron

**Species with low spatial CV (<0.3):** Poor spatial differentiation
- Couch's Kingbird: 80% temporal features, 14% spatial
- Found everywhere equally → statistics sufficient

**Conclusion:** Spatial CV is a strong pre-modeling filter for model utility.

### 3. Seasonal Models Expand Coverage

**Without seasonal models:**
- Only 58 species modelable (>10% overall detection)

**With seasonal models:**
- 90 species modelable (32 additional species rescued)
- Captures migratory and breeding season specialists

### 4. Observer Bias Removed Successfully

**Before removing effort features:**
- Duration minutes: 10-15% importance
- Effort distance: 5-8% importance
- Models learned "longer trips find more birds" (obvious, useless)

**After removal:**
- Spatial features increased to 78%
- Models learned actual habitat preferences

---

## Limitations & Future Work

### Current Limitations

**1. Observer Bias Remains:**
- Hotspots oversampled vs random locations
- Skilled birders detect more species
- Mitigation: Focused on spatial patterns within observed locations

**2. Temporal Validation Missing:**
- No hold-out test on future years
- Models trained and tested on same time period
- **Future:** Train on 2022-2024, test on 2025

**3. Spatial Cross-Validation Not Implemented:**
- Random split allows same locations in train/test
- **Future:** Leave-one-grid-cell-out validation

**4. Rare Species Excluded:**
- 341 species (67%) have <2% detection
- Unpredictable vagrants not modeled
- Trade-off: Focus on common/seasonal species

### Future Enhancements

**Methodological:**
- Spatial cross-validation to test generalization to new locations
- Temporal validation to test predictions for future dates
- Ensemble methods (gradient boosting, neural networks)
- Prediction intervals (confidence bands)

**Feature Engineering:**
- Weather data (temperature, precipitation, wind)
- Elevation (for hill country species)
- Habitat fragmentation metrics
- Distance to urban centers

**Expansion:**
- Additional counties (Travis, Cameron, Starr)
- Statewide Texas coverage
- Rare species modeling with adjusted thresholds

**Application:**
- Mobile app with GPS integration
- Personalized recommendations based on user skill level
- Route optimization (multi-location trip planning)
- Integration with photo ID apps (Merlin, iNaturalist)

---

## Technical Stack

**Languages & Libraries:**
- Python 3.x
- Pandas, NumPy (data manipulation)
- Scikit-learn (machine learning)
- GeoPandas, Rasterio, Shapely (geospatial processing)

**Data Sources:**
- eBird Basic Dataset (Cornell Lab of Ornithology)
- USGS National Land Cover Database (NLCD)
- USGS National Hydrography Dataset (NHD)
- Protected Areas Database (PAD-US)

**Deployment:**
- Streamlit (web framework)
- Streamlit Cloud (hosting)
- GitHub (version control)
- Git LFS (large model storage)

---

## Educational Value

This project demonstrates:
- **End-to-end ML pipeline:** Data cleaning → feature engineering → model training → deployment
- **Domain-specific feature engineering:** Geospatial data integration
- **Model selection strategy:** Not all problems need ML (spatial CV filtering)
- **Bias mitigation:** Identifying and removing observer behavior features
- **Calibration importance:** Predicted probabilities must match reality
- **Practical deployment:** API integration, cloud hosting, user interface

---
