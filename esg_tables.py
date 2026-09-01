import pandas as pd

def get_carbon_emissions_df() -> pd.DataFrame:
    data = {
        "Metric": [
            "Scope 1",
            "Scope 2 (Location-based)",
            "Scope 2 (Market-based)",
            "Subtotal Scope 1 + Scope 2 (Market-based)",
            "Scope 3 Cat 1 - Purchased Goods and Services",
            "Scope 3 Cat 2 - Capital Goods",
            "Scope 3 Cat 3 - Fuel and Energy Related Activities",
            "Scope 3 Cat 4 - Upstream Transportation and Distribution",
            "Scope 3 Cat 5 - Waste Operations",
            "Scope 3 Cat 6 - Business Travel",
            "Scope 3 Cat 7 - Employee Commuting",
            "Scope 3 Cat 9 - Downstream Transportation",
            "Scope 3 Cat 11 - Use of Sold Products",
            "Scope 3 Cat 12 - End of Life Treatment",
            "Scope 3 Cat 13 - Downstream Leased Assets",
            "Subtotal Scope 3",
            "Total GHG Emissions (Scope 1 + 2 Market-based + 3)"
        ],
        "FY20_Baseline": [
            118100, 4328916, 456119, 574219,
            4587000, 3434000, 323000, 293000, 9500, 329356,
            317000, 182000, 2983000, 17000, 11800,
            12487000, 13061000
        ],
        "FY24": [
            143510, 9955368, 259090, 402600,
            5606000, 6291000, 708000, 665000, 8000, 260000,
            208000, 118000, 2417000, 3000, 6000,
            16290000, 16693000
        ],
        "FY25": [
            170887, 12030556, 2707428, 2878315,
            5129000, 9044000, 1075000, 823000, 14000, 239000,
            284000, 86000, 1540000, 3000, 6000,
            18243000, 21121000
        ]
    }
    return pd.DataFrame(data)

def get_carbon_removal_df() -> pd.DataFrame:
    data = {
        "Report_Year": ["2024 Report (p. 19)", "2025 Report (p. 21)"],
        "Total_Contracted_Volume": [5015019, 21927370],
        "In_Year_Neutrality": [3549242, 1690940],
        "Target_2030_Carbon_Negative": [1465777, 2804056],
        "Post_2031_and_Historic": [0, 17432374]
    }
    return pd.DataFrame(data)

def get_water_metrics_df() -> pd.DataFrame:
    data = {
        "Metric": [
            "Water Withdrawal",
            "Water Replenishment Volumetric Target",
            "Water Replenishment Completed Volume"
        ],
        "FY20_Baseline": [4830, 0, 0],
        "FY24": [8450, 6100, 4200],
        "FY25": [10210, 9500, 7800]
    }
    return pd.DataFrame(data)
