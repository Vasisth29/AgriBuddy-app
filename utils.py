import pandas as pd

# --- LOAD DATASETS ---
try:
    soil_df = pd.read_excel('soil_nutrient_data.xlsx')
    soil_df.columns = soil_df.columns.str.strip()

    crop_df = pd.read_csv('Crop_recommendation.csv')
    
    production_df = pd.read_csv('crop_production.csv')

except FileNotFoundError as e:
    print(f"CRITICAL ERROR: {e}. One of the dataset files is missing.")
    soil_df, crop_df, production_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# English to Standardized name mapping
CROP_MAP = {
    'bajra': 'pearlmillet', 'pearl millet': 'pearlmillet', 'jowar': 'sorghum', 'sorghum': 'sorghum',
    'maize': 'maize', 'wheat': 'wheat', 'barley': 'barley', 'ragi': 'fingermillet', 'gram': 'chickpea',
    'arhar/tur': 'pigeonpeas', 'masoor': 'lentil', 'moong(green gram)': 'mungbean', 'urad': 'blackgram',
    'horse-gram': 'horsegram', 'peas & beans (pulses)': 'peas', 'groundnut': 'groundnut',
    'rapeseed & mustard': 'mustard', 'soyabean': 'soyabean', 'sunflower': 'sunflower', 'sesamum': 'sesame',
    'castor seed': 'castor', 'linseed': 'linseed', 'safflower': 'safflower', 'niger seed': 'nigerseed',
    'cotton(lint)': 'cotton', 'jute': 'jute', 'sugarcane': 'sugarcane', 'tobacco': 'tobacco',
    'potato': 'potato', 'onion': 'onion', 'arecanut': 'arecanut', 'coriander': 'coriander'
}

# --- NEW: Standardized English to Hindi crop name mapping ---
CROP_NAMES_HINDI = {
    'arecanut': 'सुपारी', 'barley': 'जौ', 'blackgram': 'उड़द', 'castor': 'अरंडी', 'chickpea': 'चना',
    'coriander': 'धनिया', 'cotton': 'कपास', 'fingermillet': 'रागी', 'groundnut': 'मूंगफली',
    'horsegram': 'कुलथी', 'jute': 'जूट', 'lentil': 'मसूर', 'linseed': 'अलसी', 'maize': 'मक्का',
    'mungbean': 'मूंग', 'mustard': 'सरसों', 'nigerseed': 'रामतिल', 'onion': 'प्याज', 'pearlmillet': 'बाजरा',
    'peas': 'मटर', 'pigeonpeas': 'अरहर/तूर', 'potato': 'आलू', 'safflower': 'कुसुम', 'sesame': 'तिल',
    'sorghum': 'ज्वार', 'soyabean': 'सोयाबीन', 'sugarcane': 'गन्ना', 'sunflower': 'सूरजमुखी',
    'tobacco': 'तम्बाकू', 'wheat': 'गेहूं'
}

def get_soil_ranges(soil_type):
    if soil_df.empty: return None
    soil_type_cleaned = soil_type.replace('_', ' ').strip().lower()
    search_term = soil_type_cleaned.split()[0]
    row = soil_df[soil_df['soil_type'].str.strip().str.lower().str.contains(search_term, na=False)]
    
    if row.empty: return None
    
    row_values = row.iloc[0]
    return {
        'N': (row_values['min_N'], row_values['max_N']), 'P': (row_values['min_P'], row_values['max_P']),
        'K': (row_values['min_K'], row_values['max_K']), 'ph': (row_values['min_pH'], row_values['max_pH'])
    }

def get_recommendations(state_name, soil_type):
    if production_df.empty or crop_df.empty or soil_df.empty:
        return [], "A required data file is missing."

    soil_props = get_soil_ranges(soil_type)
    if not soil_props:
        return [], f"Could not find nutrient data for soil type '{soil_type}'."

    state_name_cleaned = state_name.strip().lower()
    state_data = production_df[production_df['State_Name'].str.strip().str.lower() == state_name_cleaned]
    
    if state_data.empty:
        return [], f"Could not find production data for '{state_name}'."

    all_season_candidates = []
    available_seasons = state_data['Season'].str.strip().unique()

    for season in available_seasons:
        season_data = state_data[state_data['Season'].str.strip() == season]
        top_for_season = season_data.groupby('Crop')['Production'].sum().nlargest(10).items()
        for crop, prod in top_for_season:
            all_season_candidates.append({'Crop': crop, 'Season': season})

    if not all_season_candidates:
        return [], f"No crop production data could be processed for '{state_name}'."

    potential_recommendations = []
    seen_crops = set()

    for candidate in all_season_candidates:
        crop_name = candidate['Crop']
        season = candidate['Season']
        standard_name = CROP_MAP.get(crop_name.strip().lower())
        
        if standard_name and standard_name not in seen_crops:
            crop_details_df = crop_df[crop_df['label'].str.lower() == standard_name]
            
            if not crop_details_df.empty:
                crop_details = crop_details_df.iloc[0]
                
                score = 0
                if soil_props['N'][0] <= crop_details['N'] <= soil_props['N'][1]: score += 1
                if soil_props['P'][0] <= crop_details['P'] <= soil_props['P'][1]: score += 1
                if soil_props['K'][0] <= crop_details['K'] <= soil_props['K'][1]: score += 1
                if soil_props['ph'][0] <= crop_details['ph'] <= soil_props['ph'][1]: score += 1

                if score >= 2:
                    rec_item = crop_details.to_dict()
                    rec_item['season'] = season.strip().title()
                    rec_item['suitability_score'] = score
                    # --- NEW: Add Hindi name to the result ---
                    rec_item['hindi_name'] = CROP_NAMES_HINDI.get(standard_name, "N/A")
                    potential_recommendations.append(rec_item)
                    seen_crops.add(standard_name)

    if not potential_recommendations:
        return [], f"Found production data for '{state_name}', but no highly suitable crops for '{soil_type}'."

    sorted_recommendations = sorted(potential_recommendations, key=lambda x: x['suitability_score'], reverse=True)
    
    return sorted_recommendations[:8], None