import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import requests

EBIRD_API_KEY = st.secrets["EBIRD_API_KEY"] 

def get_recent_sightings(county, days=7):
    """Get recent notable sightings from eBird API"""
    
    # County codes for eBird API
    county_codes = {
        'Bexar': 'US-TX-029',
        'Hidalgo': 'US-TX-215'
    }
    
    county_code = county_codes.get(county)
    if not county_code:
        return None, None
    
    # Get county-wide notable sightings
    url = f"https://api.ebird.org/v2/data/obs/{county_code}/recent/notable"
    
    headers = {
        'X-eBirdApiToken': EBIRD_API_KEY
    }
    
    params = {
        'back': days,
        'detail': 'simple'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        county_sightings = response.json()
    except Exception as e:
        st.error(f"Error fetching county data: {e}")
        county_sightings = []
    
    # Get hotspot-specific sightings
    hotspot_sightings = {}
    
    for hotspot_name, hotspot_id in HOTSPOTS[county].items():
        url = f"https://api.ebird.org/v2/data/obs/{hotspot_id}/recent"
        params = {'back': days}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            hotspot_sightings[hotspot_name] = response.json()
        except:
            hotspot_sightings[hotspot_name] = []
    
    return county_sightings, hotspot_sightings

# Page config
st.set_page_config(
    page_title="Texas Birding Trip Planner",
    page_icon="🦅",
    layout="wide"
)

# Load species classification
@st.cache_data
def load_species_classification():
    return pd.read_csv('species_classification.csv')

# Load model performance
@st.cache_data
def load_model_performance():
    return pd.read_csv('models/metadata/model_performance_summary.csv')

# Load a specific model
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Get season from month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Sanitize filename
def sanitize_filename(name):
    return name.replace('/', '-').replace('\\', '-').replace(':', '-').replace(' ', '_')

# Representative locations for each county (centroids or popular birding spots)
COUNTY_LOCATIONS = {
    'Bexar': {
        'name': 'Central Bexar County',
        'lat': 29.45,
        'lon': -98.50,
        'grid_cell': 12,
        'dist_to_water': 5.0,
        'land_cover': 'Urban',
        'in_protected_area': 0
    },
    'Hidalgo': {
        'name': 'Central Hidalgo County',
        'lat': 26.20,
        'lon': -98.23,
        'grid_cell': 12,
        'dist_to_water': 3.0,
        'land_cover': 'Grassland',
        'in_protected_area': 0
    }
}

# ebird hotspot locations
HOTSPOTS = {
    'Bexar': {
        'Mitchell Lake Audubon Center': 'L160563',
        'Friedrich Wilderness Park': 'L160560',
        'Government Canyon': 'L355335',
        'OP Schnabel Park': 'L443189'
    },
    'Hidalgo': {
        'Estero Llano Grande State Park': 'L259855',
        'Bentsen-Rio Grande Valley State Park': 'L128890',
        'Santa Ana Wildlife Refuge': 'L129085',
        'Edinburg Scenic Wetlands': 'L249889'
    }
}

# Get predictions for all species in a county
def get_all_predictions(county, season, hour):
    # Get county location
    loc = COUNTY_LOCATIONS[county]
    
    # Determine model directory
    if county == "Bexar":
        model_dirs = ['models/bexar_only', 'models/combined']
    else:
        model_dirs = ['models/hidalgo_only', 'models/combined']
    
    predictions = []
    
    # Try to load models for this season
    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            continue
            
        for model_file in os.listdir(model_dir):
            if not model_file.endswith('.pkl'):
                continue
            
            # Check if this is the right season
            if not model_file.lower().endswith(f"{season.lower()}.pkl"):
                continue
            
            model_path = os.path.join(model_dir, model_file)
            
            try:
                # Load model
                model_package = load_model(model_path)
                model = model_package['model']
                features = model_package['features']
                species_name = model_package['species_name']
                
                # Create feature vector
                feature_dict = {
                    'LATITUDE': loc['lat'],
                    'LONGITUDE': loc['lon'],
                    'grid_cell': loc['grid_cell'],
                    'dist_to_water': loc['dist_to_water'],
                    'land_cover_Urban': 1 if loc['land_cover'] == 'Urban' else 0,
                    'land_cover_Forest': 1 if loc['land_cover'] == 'Forest' else 0,
                    'land_cover_Grassland': 1 if loc['land_cover'] == 'Grassland' else 0,
                    'land_cover_Wetland': 1 if loc['land_cover'] == 'Wetland' else 0,
                    'land_cover_Agricultural': 1 if loc['land_cover'] == 'Agricultural' else 0,
                    'land_cover_Other': 1 if loc['land_cover'] == 'Other' else 0,
                    'in_protected_area': loc['in_protected_area'],
                    'hour': hour,
                    'COUNTY_BEXAR': 1 if county == "Bexar" else 0,
                    'COUNTY_HIDALGO': 1 if county == "Hidalgo" else 0
                }
                
                X = pd.DataFrame([feature_dict])[features]
                
                # Predict
                probability = model.predict_proba(X)[0, 1]
                
                predictions.append({
                    'species_name': species_name,
                    'probability': probability,
                    'detection_rate': model_package.get('detection_rate', 0),
                    'roc_auc': model_package.get('roc_auc', 0)
                })
                
            except Exception as e:
                continue
    
    return pd.DataFrame(predictions)

# Main app
def main():
    st.title("Texas Birding Trip Planner")
    st.markdown("### Plan your birding trip - discover which species you're most likely to see")
    
    # Load data
    species_df = load_species_classification()
    performance_df = load_model_performance()
    
    # Sidebar - Trip Parameters
    st.sidebar.header("Plan Your Trip")
    
    # County selector
    county = st.sidebar.selectbox(
        "Where are you going?",
        ["Bexar", "Hidalgo"],
        help="Select the county you plan to visit"
    )
    
    # Date selector
    selected_date = st.sidebar.date_input(
        "When are you going?",
        value=datetime.now()
    )
    
    season = get_season(selected_date.month)
    st.sidebar.info(f"Season: **{season}**")
    
    # Time selector
    hour = st.sidebar.slider(
        "What time will you go birding?",
        min_value=5,
        max_value=19,
        value=8,
        format="%d:00",
        help="Most birds are active early morning (6-9 AM)"
    )
    
    # Get predictions button
    if st.sidebar.button("Find Birds", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing bird predictions for {county} County in {season}..."):
            
            # Get all predictions
            predictions_df = get_all_predictions(county, season, hour)
            
            if len(predictions_df) == 0:
                st.error(f"No models available for {county} County in {season}")
                return
            
            # Sort by probability
            predictions_df = predictions_df.sort_values('probability', ascending=False)
            
            # Summary metrics
            st.header(f"Trip Summary: {county} County - {season}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Species Predicted",
                    len(predictions_df),
                    help="Number of species with models for this season"
                )
            
            with col2:
                high_prob = len(predictions_df[predictions_df['probability'] >= 0.25])
                st.metric(
                    "High Probability (>25%)",
                    high_prob,
                    help="Species you're very likely to see"
                )
            
            with col3:
                avg_prob = predictions_df['probability'].mean()
                st.metric(
                    "Average Probability",
                    f"{avg_prob:.1%}",
                    help="Average detection probability across all species"
                )
            
            with col4:
                st.metric(
                    "Best Time",
                    f"{hour}:00",
                    help="Your selected birding time"
                )
            
            # Top species section
            st.header("Top Species to Target")
            st.markdown(f"**Most likely birds to see in {county} County during {season} at {hour}:00**")
            
            top_20 = predictions_df.head(20).copy()
            
            # Add recommendation tier
            def get_tier(prob):
                if prob >= 0.30:
                    return "Excellent"
                elif prob >= 0.20:
                    return "Very Good"
                elif prob >= 0.10:
                    return "Good"
                else:
                    return "Moderate"
            
            top_20['recommendation'] = top_20['probability'].apply(get_tier)
            
            # Display as dataframe
            display_df = top_20[['species_name', 'probability', 'recommendation', 'roc_auc']].copy()
            display_df.columns = ['Species', 'Detection Probability', 'Recommendation', 'Model Quality']
            display_df['Detection Probability'] = display_df['Detection Probability'].apply(lambda x: f"{x:.1%}")
            display_df['Model Quality'] = display_df['Model Quality'].apply(lambda x: f"{x:.3f}")
            display_df.index = range(1, len(display_df) + 1)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=600
            )
            
            # County specialties
            st.header("County Specialties")
            st.markdown(f"**Species particularly associated with {county} County:**")
            
            specialties = species_df[
                (species_df['county_specialty'] == county) |
                ((species_df['county_specialty'] == county) & (species_df['model_strategy'] != 'skip'))
            ]['species_name'].tolist()
            
            specialty_predictions = predictions_df[predictions_df['species_name'].isin(specialties)]
            
            if len(specialty_predictions) > 0:
                specialty_top = specialty_predictions.head(10)
                
                for idx, row in specialty_top.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{row['species_name']}**")
                    with col2:
                        st.metric("Probability", f"{row['probability']:.1%}")
            else:
                st.info(f"No county specialty species have high predictions for this season/time.")
            
            
            # Recent sightings section
            st.header("Recent Sightings")
            
            with st.spinner("Fetching recent eBird data..."):
                county_sightings, hotspot_sightings = get_recent_sightings(county, days=7)
                
                # Notable sightings tab
                st.subheader("Notable Sightings (Last 7 Days)")
                st.markdown(f"**Rare or unusual birds reported in {county} County:**")
                
                if county_sightings and len(county_sightings) > 0:
                    sightings_data = []
                    for obs in county_sightings[:15]:
                        sightings_data.append({
                            'Species': obs['comName'],
                            'Location': obs['locName'],
                            'Date': obs['obsDt'],
                            'Count': obs.get('howMany', 'X')
                        })
                    
                    sightings_df = pd.DataFrame(sightings_data)
                    sightings_df['In Your List'] = sightings_df['Species'].isin(
                        predictions_df['species_name']
                    ).apply(lambda x: '✅' if x else '')
                    
                    st.dataframe(sightings_df, use_container_width=True, height=400)
                    
                    # Highlight overlap
                    overlap = set(sightings_df['Species']) & set(predictions_df['species_name'])
                    if len(overlap) > 0:
                        st.success(f"**{len(overlap)} species** from recent notable sightings match your predictions!")
                else:
                    st.info("No recent notable sightings.")
                
                # Hotspot activity
                st.subheader("Activity at Top Hotspots")
                st.markdown("**What's being seen at popular birding locations:**")
                
                for hotspot_name, sightings in hotspot_sightings.items():
                    with st.expander(f"{hotspot_name} ({len(sightings)} species)"):
                        if len(sightings) > 0:
                            # Get top 10 most recent
                            hotspot_data = []
                            for obs in sightings[:10]:
                                hotspot_data.append({
                                    'Species': obs['comName'],
                                    'Date': obs['obsDt'],
                                    'Count': obs.get('howMany', 'X')
                                })
                            
                            hotspot_df = pd.DataFrame(hotspot_data)
                            hotspot_df['Match'] = hotspot_df['Species'].isin(
                                predictions_df['species_name']
                            ).apply(lambda x: '✅' if x else '')
                            
                            st.dataframe(hotspot_df, use_container_width=True)
                            
                            # Show matching predictions
                            matches = set(hotspot_df['Species']) & set(predictions_df['species_name'])
                            if len(matches) > 0:
                                st.info(f"**{len(matches)} matches** with your predictions: {', '.join(sorted(matches)[:5])}")
                        else:
                            st.write("No recent reports")
            
            # Probability distribution
            st.header("Detection Probability Distribution")
            st.markdown("Overview of all species predictions:")
            
            prob_bins = pd.cut(
                predictions_df['probability'],
                bins=[0, 0.10, 0.20, 0.30, 1.0],
                labels=['<10%', '10-20%', '20-30%', '>30%']
            )
            
            bin_counts = prob_bins.value_counts().sort_index()
            
            st.bar_chart(bin_counts)
            
            # Tips section
            st.header("Birding Tips")
            
            tip_col1, tip_col2 = st.columns(2)
            
            with tip_col1:
                st.markdown("""
                **Best Practices:**
                - Early morning (6-9 AM) is typically best
                - Bring binoculars and field guide
                - Visit multiple habitat types
                - Check recent eBird sightings
                - Respect private property and wildlife
                """)
            
            with tip_col2:
                if county == "Bexar":
                    st.markdown("""
                    **Top Bexar County Locations:**
                    - Mitchell Lake Audubon Center
                    - Government Canyon State Natural Area
                    - Friedrich Wilderness Park
                    - Eisenhower Park
                    - Phil Hardberger Park
                    """)
                else:
                    st.markdown("""
                    **Top Hidalgo County Locations:**
                    - Estero Llano Grande State Park
                    - Bentsen-Rio Grande Valley State Park
                    - Santa Ana National Wildlife Refuge
                    - Frontera Audubon
                    - Anzalduas Park
                    """)
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.markdown(f"""
    This tool uses machine learning models trained on {len(performance_df)} species-season combinations from eBird data (2022-2025).
    
    **Model Performance:**
    - Average accuracy: {performance_df['roc_auc'].mean():.1%}
    - Based on spatial habitat features
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Data: eBird | Models: Random Forest + Calibration")

if __name__ == "__main__":
    main()
