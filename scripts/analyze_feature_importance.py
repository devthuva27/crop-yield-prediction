
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_feature_importance():
    # Load feature importance
    df = pd.read_csv('../results/xgboost_feature_importance.csv')
    
    # Take top 10 for visualization
    top_10 = df.head(10).copy()
    
    # Convert importance to percentage for readability
    top_10['Importance_Pct'] = top_10['Importance'] * 100
    
    # --- 1. Visualization ---
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart with different colors for each bar
    # Using a palette with enough colors
    palette = sns.color_palette("husl", len(top_10))
    
    bars = plt.barh(top_10['Feature'], top_10['Importance_Pct'], color=palette)
    plt.xlabel('Importance (%)')
    plt.title('Top 10 Factors Affecting Crop Yield')
    plt.gca().invert_yaxis() # Highest importance at top
    
    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../results/feature_importance_top10.png')
    plt.close()
    print("Visualization saved to results/feature_importance_top10.png")

    # --- 2. Detailed Interpretation Document ---
    
    # Metadata for interpretation (Knowledge Knowledge Base)
    # We map feature substrings to explanations since names might vary slightly or be specific categories
    
    feature_meta = {
        "Crop_Sugarcane": {
            "name": "Crop Type: Sugarcane",
            "meaning": "Indicates if the crop grown is Sugarcane.",
            "why": "Sugarcane is a C4 biomass giant. It naturally produces massive tonnage (cane weight) compared to grains or spices. This is the single strongest predictor because its baseline yield is 10-20x other crops.",
            "example": "Sugarcane yields ~60,000 kg/ha. In contrast, Cinnamon yields ~1,000 kg/ha. Knowing the crop is Sugarcane immediately shifts the expected yield prediction by tens of thousands of kg."
        },
        "Crop_Rice": {
            "name": "Crop Type: Rice",
            "meaning": "Indicates if the crop grown is Rice (Paddy).",
            "why": "Rice is a staple grain with moderate biomass. It requires flooded conditions (paddy) and has a very specific yield range, distinct from tree crops or spices.",
            "example": "Rice typically yields 3,000-5,000 kg/ha. The model uses this to distinguish it from high-yielding Sugarcane or lower-yielding spices."
        },
        "District_Galle": {
            "name": "District: Galle",
            "meaning": "Farm location is in the Galle District (Southern Province, Wet Zone).",
            "why": "Galle receives high rainfall and has specific soil types (red-yellow podzolic). It is a hub for export crops like Cinnamon, Tea, and Rubber. Location here acts as a proxy for these specific cropping systems and climate stability.",
            "example": "A farm in Galle is likely to be a wet-zone perennial crop (like Cinnamon or Tea) rather than dry-zone seasonal crops, influencing the expected yield consistency and magnitude."
        },
        "District_Matara": {
            "name": "District: Matara",
            "meaning": "Farm location is in the Matara District.",
            "why": "Adjacent to Galle, Matara is a key agricultural region in the Wet Zone, known specifically for high-quality Cinnamon and Paddy cultivation. It represents a specific agro-ecological zone.",
            "example": "Matara has optimal conditions for Cinnamon. Finding a farm here increases the probability of it being a spice crop or high-yield paddy vs. a dry zone crop."
        },
        "District_Kalutara": {
            "name": "District: Kalutara",
            "meaning": "Farm location is in the Kalutara District (Western Province).",
            "why": "Kalutara is famous for Rubber and Mangosteen, as well as Rice. It is very wet. The model likely associates this district with specific yield profiles of Rubber or Rice.",
            "example": "Rubber plantations in Kalutara yield latex/timber. The yield metrics for Rubber (latex) are distinct from food crops. This feature helps categorize the farm's potential output."
        },
        "Rainfall_mm": {
            "name": "Rainfall (mm)",
            "meaning": "Total precipitation during the growing season.",
            "why": "Water availability limits growth. In Sri Lanka, variability in monsoons affects yield. More rain (up to a point) generally supports higher biomass.",
            "example": "An increase from 50mm to 200mm rainfall can significantly boost vegetative growth in water-hungry crops like Rice."
        },
        "rainfall_to_temperature_ratio": {
            "name": "Rainfall/Temp Ratio",
            "meaning": "A measure of moisture availability relative to heat stress (Aridity Index proxy).",
            "why": "It's not just rain; it's how fast it evaporates. A higher ratio means cool/moist conditions favorable for most wet-zone crops. Low ratio means hot/dry stress.",
            "example": "A ratio of 2.0 (wet/cool) supports lush growth, while 0.5 (dry/hot) indicates stress that reduces yield per hectare."
        }
    }
    
    top_5 = df.head(5)
    
    markdown_content = "# Detailed Feature Importance Analysis (XGBoost)\n\n"
    markdown_content += "This document breaks down the top 5 drivers of crop yield based on the best-performing model (XGBoost).\n\n"
    
    rank = 1
    for _, row in top_5.iterrows():
        feat = row['Feature']
        imp = row['Importance'] * 100
        
        # Look up metadata
        meta = feature_meta.get(feat, {
            "name": feat,
            "meaning": "Specific feature from dataset.",
            "why": "Identified by the model as a key split point for yield prediction.",
            "example": f"Changes in {feat} result in significant yield variance."
        })
        
        markdown_content += f"## #{rank} - {meta['name']} ({imp:.1f}% importance)\n\n"
        markdown_content += f"**Rank:** #{rank}\n"
        markdown_content += f"**Importance Score:** {imp:.1f}%\n"
        markdown_content += f"**Feature Name:** {feat}\n\n"
        markdown_content += f"**Agricultural Meaning:**\n{meta['meaning']}\n\n"
        markdown_content += f"**Why It Matters:**\n{meta['why']}\n\n"
        markdown_content += f"**Real-World Example:**\n{meta['example']}\n\n"
        markdown_content += "---\n\n"
        rank += 1
        
    with open('../results/feature_importance_detailed.md', 'w') as f:
        f.write(markdown_content)
    print("Detailed report saved to results/feature_importance_detailed.md")

    # --- 3. Summary Document ---
    
    summary_content = "Feature Importance Summary\n"
    summary_content += "===========================\n\n"
    
    # Analyze the dominance
    top_feat = df.iloc[0]
    top_feat_name = top_feat['Feature']
    top_feat_imp = top_feat['Importance']
    
    summary_content += "1. Agricultural Domain Insights\n"
    summary_content += "--------------------------------\n"
    summary_content += "The analysis reveals that **Crop Type** is the overwhelming determinant of Yield (Mass/Area). \n"
    summary_content += f"Specifically, '{top_feat_name}' alone explains {top_feat_imp:.1%} of the variance. \n"
    summary_content += "This is agriculturally intuitive: you cannot compare the biomass of a grass like Sugarcane (tonnes/ha) \n"
    summary_content += "directly with the biomass of a spice like Cinnamon (kg/ha) without this factor being dominant.\n\n"
    
    summary_content += "2. What does the model tell us about farming?\n"
    summary_content += "------------------------------------------\n"
    summary_content += "- **Biology First:** The genetic potential of the crop (Species) sets the yield ceiling/floor more than any management practice (fertilizer) or weather event.\n"
    summary_content += "- **Location Matters:** After crop type, the specific Districts (Galle, Matara, Kalutara) are the next most critical factors. This implies that soil, local microclimates, or regional expertise play a huge role.\n"
    summary_content += "  - Interestingly, these are all Wet Zone districts. The model finds the Wet vs. Dry zone distinction crucial.\n"
    summary_content += "- **Weather & Inputs are Secondary:** In this mixed-crop dataset, Rainfall and Nutrients have lower relative importance scores. This doesn't mean they don't matter; it means *Crop Type* masks their effects. E.g., Rice needs fertilizer, but upgrading from Rice to Sugarcane changes yield by 10x, whereas fertilizer changes it by 10%.\n\n"
    
    summary_content += "3. Do results match agricultural knowledge?\n"
    summary_content += "------------------------------------------\n"
    summary_content += "Yes. In a multi-crop dataset, species differences are always the primary variance driver. \n"
    summary_content += "If we trained a model *only* on Rice, we would see Nitrogen and Rainfall jump to the top. \n"
    summary_content += "But globally across crops, 'What did you plant?' is more important than 'How much did it rain?'.\n\n"
    
    summary_content += "4. Surprising Findings\n"
    summary_content += "----------------------\n"
    summary_content += "- The massive dominance of Sugarcane (85%+) suggests the dataset might be skewed or Sugarcane yields are outliers in magnitude.\n"
    summary_content += "- 'Nutrient' features (Nitrogen, P, K) are lower in the list (outside top 5). This is surprising for agronomy but explainable by the scale difference between crops.\n"
    
    with open('../results/feature_importance_summary.txt', 'w') as f:
        f.write(summary_content)
    print("Summary saved to results/feature_importance_summary.txt")

if __name__ == "__main__":
    analyze_feature_importance()
