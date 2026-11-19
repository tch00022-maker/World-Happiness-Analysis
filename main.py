
# WORLD HAPPINESS REPORT ANALYSIS (2015-2019)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pathlib import Path
from scipy import stats

# --------------------------------------------------
# SETUP
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = BASE_DIR / "images"

IMAGE_DIR.mkdir(exist_ok=True)

years = [2015, 2016, 2017, 2018, 2019]

sns.set_theme(style="whitegrid")

# --------------------------------------------------
# LOAD AND STANDARDIZE DATASETS
# --------------------------------------------------

data = {}

# Different names used for the same variables
column_aliases = {
    "Country": ["country", "countryorregion"],
    "Region": ["region"],
    "Score": ["score", "happinessscore"],
    "Rank": ["rank", "happinessrank", "overallrank"],
    "GDP": [
        "gdppercapita",
        "economygdppercapita",
        "economygdppercapita"
    ],
    "Health": [
        "health",
        "healthlifeexpectancy",
        "healthylifeexpectancy"
    ],
    "Freedom": [
        "freedom",
        "freedomtomakelifechoices"
    ],
    "Generosity": ["generosity"],
    "Corruption": [
        "corruption",
        "trustgovernmentcorruption",
        "perceptionsofcorruption"
    ],
    "Social Support": [
        "socialsupport",
        "family"
    ]
}


def normalize_column(name):
    """Remove spaces and punctuation from column names."""
    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )


def standardize_dataset(df):
    """Rename different yearly column formats consistently."""

    rename = {}

    for standard_name, aliases in column_aliases.items():
        for column in df.columns:
            normalized = normalize_column(column)

            if normalized in aliases:
                rename[column] = standard_name
                break

    df = df.rename(columns=rename)

    # Make sure all required columns exist
    required_columns = [
        "Country", "Region", "Score", "Rank",
        "GDP", "Health", "Freedom",
        "Generosity", "Corruption", "Social Support"
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = np.nan

    # Convert numerical columns
    numerical_columns = [
        "Score", "Rank", "GDP", "Health",
        "Freedom", "Generosity", "Corruption",
        "Social Support"
    ]

    for column in numerical_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    return df[required_columns].copy()


print("\nWORLD HAPPINESS REPORT ANALYSIS")
print("=" * 60)

for year in years:

    file_path = DATA_DIR / f"{year}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. "
            "Check your data folder and filenames."
        )

    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()

    df = standardize_dataset(df)

    df["Year"] = year

    data[year] = df

    print(f"\n{year} dataset loaded successfully")
    print("Countries:", len(df))
    print("Columns:", df.columns.tolist())

print("\nAll datasets loaded and standardized!")

# --------------------------------------------------
# QUESTION 1
# DISTRIBUTION OF HAPPINESS SCORES
# --------------------------------------------------

print("\nQUESTION 1: HAPPINESS SCORE DISTRIBUTIONS")

for year in years:

    df = data[year]

    plt.figure(figsize=(9, 5))

    sns.histplot(
        df["Score"].dropna(),
        bins=15,
        kde=True
    )

    plt.title(f"Happiness Score Distribution - {year}")
    plt.xlabel("Happiness Score")
    plt.ylabel("Number of Countries")

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / f"Q1_happiness_distribution_{year}.png",
        dpi=300
    )

    plt.close()

    print(f"{year}: Mean happiness score = "
          f"{df['Score'].mean():.3f}")

print("Question 1 completed!")

# --------------------------------------------------
# QUESTION 2
# GDP PER CAPITA VS HEALTH
# --------------------------------------------------

print("\nQUESTION 2: GDP PER CAPITA VS HEALTH")

for year in years:

    df = data[year].dropna(
        subset=["GDP", "Health"]
    )

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="GDP",
        y="Health"
    )

    for _, row in df.iterrows():
        plt.annotate(
            row["Country"],
            (row["GDP"], row["Health"]),
            fontsize=5,
            alpha=0.6
        )

    plt.title(f"GDP per Capita vs Health - {year}")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Health / Life Expectancy")

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / f"Q2_GDP_vs_Health_{year}.png",
        dpi=300
    )

    plt.close()

    correlation = df["GDP"].corr(df["Health"])

    print(f"{year}: Pearson correlation = "
          f"{correlation:.3f}")

print("Question 2 completed!")

# --------------------------------------------------
# QUESTION 3
# FREEDOM VS GENEROSITY
# --------------------------------------------------

print("\nQUESTION 3: FREEDOM VS GENEROSITY")

for year in years:

    df = data[year].dropna(
        subset=["Freedom", "Generosity"]
    )

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="Freedom",
        y="Generosity"
    )

    for _, row in df.iterrows():
        plt.annotate(
            row["Country"],
            (row["Freedom"], row["Generosity"]),
            fontsize=5,
            alpha=0.6
        )

    plt.title(f"Freedom vs Generosity - {year}")
    plt.xlabel("Freedom")
    plt.ylabel("Generosity")

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / f"Q3_Freedom_vs_Generosity_{year}.png",
        dpi=300
    )

    plt.close()

    correlation = df["Freedom"].corr(df["Generosity"])

    print(f"{year}: Pearson correlation = "
          f"{correlation:.3f}")

print("Question 3 completed!")

# --------------------------------------------------
# QUESTION 4
# THREE-DIMENSIONAL SCATTER PLOT
# GDP, CORRUPTION, GENEROSITY
# --------------------------------------------------

print("\nQUESTION 4: THREE-DIMENSIONAL SCATTER PLOTS")

for year in years:

    df = data[year].dropna(
        subset=["GDP", "Corruption", "Generosity"]
    )

    fig = plt.figure(figsize=(11, 8))

    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        df["GDP"],
        df["Corruption"],
        df["Generosity"],
        c=df["Generosity"],
        cmap="viridis",
        s=35
    )

    ax.set_title(
        f"GDP, Corruption and Generosity - {year}"
    )

    ax.set_xlabel("GDP per Capita")
    ax.set_ylabel("Corruption")
    ax.set_zlabel("Generosity")

    fig.colorbar(
        scatter,
        ax=ax,
        label="Generosity"
    )

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / f"Q4_3D_scatter_{year}.png",
        dpi=300
    )

    plt.close(fig)

    print(f"{year}: 3D scatter plot created.")

print("Question 4 completed!")

# --------------------------------------------------
# QUESTION 5
# COUNTRIES MISSING BETWEEN CONSECUTIVE YEARS
# --------------------------------------------------

print("\nQUESTION 5: MISSING COUNTRIES")

for i in range(len(years) - 1):

    year1 = years[i]
    year2 = years[i + 1]

    countries1 = set(
        data[year1]["Country"].dropna().str.strip()
    )

    countries2 = set(
        data[year2]["Country"].dropna().str.strip()
    )

    missing_from_year2 = sorted(countries1 - countries2)
    missing_from_year1 = sorted(countries2 - countries1)

    print(f"\nComparing {year1} and {year2}")

    print(f"Countries in {year1} but not {year2}:")
    print(missing_from_year2)

    print(f"Countries in {year2} but not {year1}:")
    print(missing_from_year1)

print("Question 5 completed!")

# --------------------------------------------------
# QUESTION 6
# PEARSON, SPEARMAN AND KENDALL CORRELATIONS
# --------------------------------------------------

print("\nQUESTION 6: CORRELATION COEFFICIENTS")

correlation_pairs = [
    ("Score", "Generosity"),
    ("Freedom", "Generosity"),
    ("GDP", "Corruption"),
    ("GDP", "Freedom")
]

methods = [
    ("Pearson", "pearson"),
    ("Spearman", "spearman"),
    ("Kendall", "kendall")
]

for year in years:

    df = data[year]

    print(f"\n{'=' * 45}")
    print(f"CORRELATIONS FOR {year}")
    print("=" * 45)

    for variable1, variable2 in correlation_pairs:

        subset = df[[variable1, variable2]].dropna()

        print(f"\n{variable1} vs {variable2}")

        if len(subset) < 2:
            print("Not enough data.")
            continue

        for method_name, method in methods:

            result = subset[variable1].corr(
                subset[variable2],
                method=method
            )

            print(f"{method_name}: {result:.4f}")

print("Question 6 completed!")

# --------------------------------------------------
# QUESTION 7
# VARIABLES MOST STRONGLY RELATED TO HAPPINESS
# --------------------------------------------------

print("\nQUESTION 7: STRONGEST HAPPINESS RELATIONSHIPS")

happiness_variables = [
    "GDP",
    "Health",
    "Freedom",
    "Generosity",
    "Corruption",
    "Social Support"
]

for year in years:

    df = data[year]

    print(f"\nHAPPINESS CORRELATIONS: {year}")

    results = []

    for variable in happiness_variables:

        subset = df[["Score", variable]].dropna()

        if len(subset) < 2:
            continue

        correlation = subset["Score"].corr(
            subset[variable]
        )

        results.append(
            (variable, correlation)
        )

    results.sort(
        key=lambda item: abs(item[1]),
        reverse=True
    )

    for variable, correlation in results:
        print(f"{variable}: {correlation:.4f}")

    if results:
        strongest = results[0]

        print(
            f"\nStrongest relationship: "
            f"{strongest[0]} ({strongest[1]:.4f})"
        )

print("Question 7 completed!")

# --------------------------------------------------
# QUESTION 8
# WORLD HAPPINESS MAPS
# --------------------------------------------------

print("\nQUESTION 8: WORLD HAPPINESS MAPS")

for year in years:

    df = data[year].dropna(
        subset=["Country", "Score"]
    )

    fig = px.choropleth(
        df,
        locations="Country",
        locationmode="country names",
        color="Score",
        hover_name="Country",
        color_continuous_scale="RdYlGn",
        title=f"World Happiness Scores - {year}",
        labels={"Score": "Happiness Score"}
    )

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth"
        )
    )

    # Save interactive map
    fig.write_html(
        str(IMAGE_DIR / f"Q8_world_map_{year}.html")
    )

    # Save static image when Kaleido is available
    try:
        fig.write_image(
            str(IMAGE_DIR / f"Q8_world_map_{year}.png"),
            width=1200,
            height=700
        )

        print(f"{year}: World map saved as PNG and HTML.")

    except Exception as error:

        print(
            f"{year}: Interactive HTML map saved. "
            f"Static PNG export unavailable: {error}"
        )

print("Question 8 completed!")

# --------------------------------------------------
# QUESTION 9
# HAPPINESS RANK TRENDS BY YEAR AND REGION
# --------------------------------------------------

print("\nQUESTION 9: HAPPINESS RANK TRENDS")

# Combine all years into one dataset
all_data = pd.concat(
    [data[year] for year in years],
    ignore_index=True
)

# Average overall rank for each year
yearly_rank = (
    all_data.groupby("Year")["Rank"]
    .mean()
)

print("\nAverage Happiness Rank by Year:")
print(yearly_rank)

# Plot average rank by year
plt.figure(figsize=(10, 6))

sns.lineplot(
    x=yearly_rank.index,
    y=yearly_rank.values,
    marker="o"
)

plt.title("Average Happiness Rank by Year")
plt.xlabel("Year")
plt.ylabel("Average Happiness Rank")
plt.xticks(years)
plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "Q9_average_rank_by_year.png",
    dpi=300
)

plt.close()

# Create a country-to-region lookup using
# the years that contain regional information
region_data = all_data.dropna(
    subset=["Country", "Region"]
)

region_lookup = (
    region_data.groupby("Country")["Region"]
    .agg(lambda values: values.mode().iloc[0])
    .to_dict()
)

# Fill in missing regional information
all_data["Region"] = all_data.apply(
    lambda row: (
        row["Region"]
        if pd.notna(row["Region"])
        else region_lookup.get(row["Country"], np.nan)
    ),
    axis=1
)

# Calculate average rank by region and year
regional_rank = (
    all_data.dropna(subset=["Region", "Rank"])
    .groupby(["Year", "Region"])["Rank"]
    .mean()
    .reset_index()
)

print("\nAverage Happiness Rank by Region and Year:")
print(
    regional_rank.pivot(
        index="Region",
        columns="Year",
        values="Rank"
    ).round(2)
)

# Plot regional rank trends
plt.figure(figsize=(12, 7))

sns.lineplot(
    data=regional_rank,
    x="Year",
    y="Rank",
    hue="Region",
    marker="o"
)

plt.title("Average Happiness Rank by Region and Year")
plt.xlabel("Year")
plt.ylabel("Average Happiness Rank")
plt.xticks(years)
plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "Q9_regional_rank_trends.png",
    dpi=300
)

plt.close()

print("Question 9 completed!")

# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("WORLD HAPPINESS ANALYSIS COMPLETED!")
print("=" * 60)

print(f"\nAll available graphs and maps are saved in:")
print(IMAGE_DIR)

print("\nYears analyzed:", years)

print("\nTotal number of datasets:", len(data))


