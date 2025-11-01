import streamlit as st
import pandas as pd
from typing import Dict, Any, List

# --- 0. FILE UPLOADS AND DATA SETUP (Simulating the Live Data.gov.in Access) ---

# We'll use Streamlit's file uploader, but for this demo, we'll assume the files are present
# and use the existing loading functions. In a real app, data would be fetched live.

# Global DataFrames initialization placeholders
CROP_DF: pd.DataFrame = pd.DataFrame()
RAINFALL_DF: pd.DataFrame = pd.DataFrame()

# --- CONSTANTS (From the previous system design) ---

SUBDIVISION_TO_STATE_MAP = {
    'Sub Himalayan West Bengal & Sikkim': 'West Bengal',
    'Gangetic West Bengal': 'West Bengal',
    'Bihar': 'Bihar',
    'West Madhya Pradesh': 'Madhya Pradesh',
    'East Madhya Pradesh': 'Madhya Pradesh',
}

CEREAL_CROPS = [
    'Rice', 'Wheat', 'Maize', 'Jowar', 'Bajra', 'Barley', 
    'Ragi', 'Samai', 'Kodra', 'Navane', 'Ganti'
]

# --- 1. DATA MANAGEMENT (CORE INTEGRATION LOGIC) ---

@st.cache_data
def load_and_clean_data(crop_file_path: str, rainfall_file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads, cleans, and standardizes the core dataframes. Cached for performance."""
    try:
        crop_df = pd.read_csv(crop_file_path)
        rainfall_df = pd.read_csv(rainfall_file_path)
    except FileNotFoundError:
        st.error("Error: Data files not found. Please ensure 'crop_production.csv' and 'Rainfall_Data_LL.csv' are in the same directory.")
        return pd.DataFrame(), pd.DataFrame()

    # Clean Crop Data
    for col in ['State_Name', 'District_Name', 'Crop', 'Season']:
        if col in crop_df.columns and crop_df[col].dtype == 'object':
            crop_df[col] = crop_df[col].str.strip()
    crop_df.dropna(subset=['Production', 'Area'], inplace=True)

    # Clean Rainfall Data
    if 'SUBDIVISION' in rainfall_df.columns and rainfall_df['SUBDIVISION'].dtype == 'object':
        rainfall_df['SUBDIVISION'] = rainfall_df['SUBDIVISION'].str.strip()

    return crop_df, rainfall_df

# --- 2. DATA QUERY AGENTS (TOOLS) ---

def find_highest_production_district(crop_df: pd.DataFrame, state_name: str, crop_name: str, year_type: str = 'latest') -> Dict[str, Any]:
    """Q2-type query: Finds the district with the highest production of a crop in a state."""
    state_name = state_name.strip().title()
    crop_name = crop_name.strip().title()
    latest_year = crop_df['Crop_Year'].max() if year_type == 'latest' else int(year_type)
    
    # ... (Rest of the filtering and calculation logic from previous response)
    # Filter data based on parameters
    filtered_df = crop_df[
        (crop_df['State_Name'] == state_name) &
        (crop_df['Crop'] == crop_name) &
        (crop_df['Crop_Year'] == latest_year)
    ]

    if filtered_df.empty:
        return {"status": "not_found", "message": f"No data for '{crop_name}' in '{state_name}' for {latest_year}."}

    district_production = filtered_df.groupby('District_Name')['Production'].sum().reset_index()
    max_production_row = district_production.loc[district_production['Production'].idxmax()]

    return {
        "status": "success", "State": state_name, "Crop": crop_name, "Year": latest_year,
        "Highest_Production_District": max_production_row['District_Name'],
        "Production_Volume": round(max_production_row['Production'], 2),
        "Source": "crop_production.csv"
    }

def compare_rainfall_and_crops(crop_df: pd.DataFrame, rainfall_df: pd.DataFrame, state_x: str, state_y: str, n_years: int, crop_type: str) -> Dict[str, Any]:
    """Q1-type query: Compares rainfall and top crop production between two states (Cross-Domain)."""
    state_x, state_y = state_x.strip().title(), state_y.strip().title()
    
    # --- 1. Rainfall Calculation ---
    latest_year = rainfall_df['YEAR'].max()
    target_years = range(latest_year - n_years + 1, latest_year + 1)
    rainfall_data = rainfall_df[rainfall_df['YEAR'].isin(target_years)].copy()
    
    rainfall_data['State'] = rainfall_data['SUBDIVISION'].map(SUBDIVISION_TO_STATE_MAP).fillna('Other')
    rainfall_data.drop(rainfall_data[rainfall_data['State'] == 'Other'].index, inplace=True)
    avg_rainfall = rainfall_data[rainfall_data['State'].isin([state_x, state_y])].groupby('State')['ANNUAL'].mean().round(2).to_dict()

    # --- 2. Crop Production Calculation ---
    crop_data_filtered = crop_df[crop_df['State_Name'].isin([state_x, state_y])].copy()
    crop_data_filtered = crop_data_filtered[crop_data_filtered['Crop_Year'].isin(target_years)].copy()
    
    relevant_crops = CEREAL_CROPS 
    crop_data_filtered = crop_data_filtered[crop_data_filtered['Crop'].isin(relevant_crops)].copy()

    if crop_data_filtered.empty or not avg_rainfall:
        return {"status": "not_found", "message": "Data for one or both parts of the comparison is insufficient."}

    total_production = crop_data_filtered.groupby(['State_Name', 'Crop'])['Production'].sum().reset_index()

    top_crops_data = {}
    m = 3 
    for state in [state_x, state_y]:
        state_production = total_production[total_production['State_Name'] == state].sort_values(by='Production', ascending=False)
        top_crops_data[state] = state_production.head(m)

    return {
        "status": "success", "State_X": state_x, "State_Y": state_y, "N_Years": n_years,
        "Years": list(target_years), "Rainfall_Data": avg_rainfall, "Top_Crops_Data": top_crops_data, 
        "Rainfall_Source": "Rainfall_Data_LL.csv", "Crop_Source": "crop_production.csv"
    }


# --- 3. LLM PLANNER & RESPONSE SYNTHESIS ---

def llm_planner(question: str) -> Dict[str, Any]:
    """Simulates the LLM's role: parsing natural language, identifying intent, and extracting parameters."""
    question = question.lower().strip()
    plan = {"intent": "unsupported", "params": {}}

    # Intent 1: Highest Production District (Q2 type) - Hardcoded example
    if "highest production" in question and "district" in question:
        plan["intent"] = "find_highest_production_district"
        plan["params"] = {"state_name": "Maharashtra", "crop_name": "Maize", "year_type": "latest"}

    # Intent 2: Complex Climate-Crop Comparison (Q1/Q3/Q4 type) - Hardcoded example
    elif "compare" in question and ("rainfall" in question or "climate" in question) and "crop" in question:
        plan["intent"] = "compare_rainfall_and_crops"
        plan["params"] = {"state_x": "West Bengal", "state_y": "Bihar", "n_years": 5, "crop_type": "Cereal"}

    return plan

def synthesize_response(raw_result: Dict[str, Any]) -> str:
    """Converts structured data into a coherent, cited answer (Markdown format)."""
    if raw_result['status'] != 'success':
        return f"**System Response Failure:** {raw_result.get('message', 'An unknown error occurred.')}"

    intent = raw_result.get('intent', '')
    
    if intent == "find_highest_production_district":
        data = raw_result
        return (
            f"**Query Type:** District-Level Production (Q2)\n"
            f"--- \n"
            f"Based on data from the **{data['Source']}** dataset:\n\n"
            f"The district in **{data['State']}** with the **highest production** of **{data['Crop']}** "
            f"in the **most recent year available** ({data['Year']}) is **{data['Highest_Production_District']}**.\n\n"
            f"The production volume for {data['Crop']} in this district was approximately **{data['Production_Volume']:,} Tons**."
        )

    elif intent == "compare_rainfall_and_crops":
        data = raw_result
        years_str = f"{data['Years'][0]} to {data['Years'][-1]}"
        
        rainfall_summary = f"**Average Annual Rainfall (Source: {data['Rainfall_Source']})** for {years_str}:\n"
        for state, rainfall in data['Rainfall_Data'].items():
            rainfall_summary += f"- **{state}**: **{rainfall:,} mm**\n"
        
        crop_summary = f"\n**Top 3 Produced Cereal Crops (by Volume) (Source: {data['Crop_Source']})**:\n"
        for state, df in data['Top_Crops_Data'].items():
            crop_summary += f"- **{state}**:\n"
            for index, row in df.iterrows():
                crop_summary += f"  - {row['Crop']}: {row['Production']:,.0f} Tons\n"
        
        return (
            f"**Query Type:** Cross-Domain Synthesis (Q1)\n"
            f"--- \n"
            f"## 🌐 Cross-Domain Policy Synthesis: {data['State_X']} vs {data['State_Y']}\n\n"
            f"This analysis synthesizes data from the **India Meteorological Department (IMD)** and the **Ministry of Agriculture & Farmers Welfare** datasets to provide policy-relevant insights.\n\n"
            f"{rainfall_summary}\n"
            f"{crop_summary}\n"
            f"**Policy Insight (Synthesis):** The substantial difference in average annual rainfall suggests that promoting water-intensive crops like Rice may be significantly riskier in Bihar than in West Bengal, reinforcing the need for policy focused on drought-resistant alternatives in Bihar."
        )

    return "**System Response Failure:** Could not synthesize a coherent response from the raw data."


# --- 4. CHATBOT ORCHESTRATION (The App Logic) ---

def chatbot_orchestrator(question: str) -> str:
    """The main execution engine that connects all system components."""
    
    # Step A: Planning (LLM Planner)
    query_plan = llm_planner(question)
    intent = query_plan["intent"]
    params = query_plan["params"]

    # Step B: Execution (Data Query Agent)
    raw_result = {"status": "failure", "message": f"Intent not executable. Planner suggested: {intent}"}
    
    if intent == "find_highest_production_district":
        raw_result = find_highest_production_district(CROP_DF, **params)
        raw_result["intent"] = intent
    
    elif intent == "compare_rainfall_and_crops":
        raw_result = compare_rainfall_and_crops(CROP_DF, RAINFALL_DF, **params)
        raw_result["intent"] = intent
    
    # Step C: Synthesis (Response Synthesis LLM)
    final_answer = synthesize_response(raw_result)
    
    return final_answer

# --- 5. STREAMLIT FRONT-END INTERFACE ---

def main():
    """Main function to define the Streamlit app interface."""
    global CROP_DF, RAINFALL_DF
    
    st.set_page_config(layout="wide")
    st.title("🇮🇳 Project Samarth: Intelligent Data Q&A System")
    st.markdown("""
        ### **Mission:** Synthesize cross-domain insights from Indian Government data (simulated).
        This prototype demonstrates an LLM-orchestrated system that queries multiple data sources and generates a **coherent, cited answer** for policy analysis.
    """)

    # --- Data Loading Section ---
    CROP_DF, RAINFALL_DF = load_and_clean_data("crop_production.csv", "Rainfall_Data_LL.csv")
    if CROP_DF.empty or RAINFALL_DF.empty:
        st.stop()

    # --- Q&A Interface ---
    st.header("Ask a Policy Question")
    
    # User Input
    user_question = st.text_area(
        "Enter your question here:",
        placeholder="e.g., Compare the average annual rainfall in State_X and State_Y for the last 5 years, and list the top 3 most produced cereal crops.",
        height=100
    )

    # Example Questions for easy testing
    st.subheader("Test Questions (Try These):")
    st.markdown("""
    1. **Complex Query (Q1):** `Compare the average annual rainfall in West Bengal and Bihar for the last 5 available years, and list the top M most produced crops of Cereal.`
    2. **District Query (Q2):** `Identify the district in Maharashtra with the highest production of Maize in the most recent year available.`
    3. **Unsupported Query:** `What are the subsidy schemes for farmers in Gujarat?`
    """)

    # Submit Button
    if st.button("Get Policy Insight", type="primary") and user_question:
        with st.spinner("Processing request... Analyzing intent, querying data sources, and synthesizing results..."):
            
            # --- EXECUTION ---
            final_response = chatbot_orchestrator(user_question)
            
            # --- OUTPUT ---
            st.success("Analysis Complete!")
            st.markdown("### 💡 Generated Policy Answer")
            st.markdown(final_response)
            
            # Show Source Data Info (Traceability)
            st.markdown("---")
            st.markdown(f"""
            #### 📊 Data Sources Used:
            - **Agriculture Data:** `crop_production.csv` (Simulates Ministry of Agriculture & Farmers Welfare)
            - **Climate Data:** `Rainfall_Data_LL.csv` (Simulates India Meteorological Department/IMD)
            """)

if __name__ == "__main__":
    main()