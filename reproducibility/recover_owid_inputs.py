"""Reconstruct archived weekly inputs from a hash-verified OWID version.
Requires pandas and numpy. Run from repository root:
python reproducibility/recover_owid_inputs.py
"""
import argparse, hashlib, json, urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

def display(*args):
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, help='Use an already downloaded CSV')
parser.add_argument('--output', type=Path, default=Path('recovered_owid'))
args = parser.parse_args()
output_dir = args.output
output_dir.mkdir(parents=True, exist_ok=True)
source_file = args.source or output_dir / 'owid-covid-data.csv'
SOURCE_URL = 'https://raw.githubusercontent.com/owid/covid-19-data/f0d53320b19703cf565940d8269181a8423d7032/public/data/owid-covid-data.csv'
EXPECTED_SHA256 = '8473d0f0fdf962e1ffbd5b85b18726fc96a49bab109e271186c339725a12b10c'
if not source_file.exists():
    if args.source:
        raise FileNotFoundError(source_file)
    urllib.request.urlretrieve(SOURCE_URL, source_file)
with source_file.open('rb') as stream:
    actual_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
if actual_sha256 != EXPECTED_SHA256:
    raise ValueError(f'OWID checksum mismatch: {actual_sha256}')
import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"

df = pd.read_csv(source_file, parse_dates=["date"], low_memory=False)

df_global = df[
    df["iso_code"].notna() &
    ~df["iso_code"].str.startswith("OWID_")
].copy()

core_vars = [
    "iso_code",
    "continent",
    "location",
    "date",
    "population",
    "new_cases_per_million",
    "weekly_hosp_admissions_per_million",
    "hosp_patients_per_million",
    "icu_patients_per_million",
    "new_deaths_per_million",
    "people_vaccinated_per_hundred",
    "stringency_index"
]

df_global = df_global[[c for c in core_vars if c in df_global.columns]]

df_global = df_global.sort_values(["location", "date"]).reset_index(drop=True)

print("Countries:", df_global["location"].nunique())
print("Rows:", len(df_global))
print("Date range:", df_global["date"].min(), "to", df_global["date"].max())

df_global.head()

# STEP 2 — GLOBAL DATA COVERAGE AUDIT

coverage = pd.DataFrame({
    "cases_non_missing":
        df_global.groupby("location")["new_cases_per_million"]
        .apply(lambda x: x.notna().sum()),

    "admissions_non_missing":
        df_global.groupby("location")["weekly_hosp_admissions_per_million"]
        .apply(lambda x: x.notna().sum()),

    "occupancy_non_missing":
        df_global.groupby("location")["hosp_patients_per_million"]
        .apply(lambda x: x.notna().sum()),

    "icu_non_missing":
        df_global.groupby("location")["icu_patients_per_million"]
        .apply(lambda x: x.notna().sum()),

    "deaths_non_missing":
        df_global.groupby("location")["new_deaths_per_million"]
        .apply(lambda x: x.notna().sum())
})

coverage = coverage.reset_index()

# Add continent
continents = (
    df_global[["location", "continent"]]
    .drop_duplicates()
)

coverage = coverage.merge(
    continents,
    on="location",
    how="left"
)

# Sort by occupancy coverage
coverage = coverage.sort_values(
    "occupancy_non_missing",
    ascending=False
)

print(coverage.head(20))

# STEP 3 — DEFINE ANALYSIS TIERS

# Tier 1:
# Countries with BOTH admissions and occupancy
tier1 = coverage[
    (coverage["cases_non_missing"] >= 300) &
    (coverage["admissions_non_missing"] >= 300) &
    (coverage["occupancy_non_missing"] >= 300)
].copy()

# Tier 2:
# Countries with occupancy only
tier2 = coverage[
    (coverage["cases_non_missing"] >= 300) &
    (coverage["occupancy_non_missing"] >= 300) &
    (coverage["admissions_non_missing"] < 300)
].copy()

# Tier 3:
# Countries with ICU only
tier3 = coverage[
    (coverage["cases_non_missing"] >= 300) &
    (coverage["icu_non_missing"] >= 300)
].copy()

print("Tier 1 countries:", len(tier1))
print("Tier 2 countries:", len(tier2))
print("Tier 3 countries:", len(tier3))

print("\nTier 1 sample:")
print(
    tier1[
        ["location", "continent",
         "admissions_non_missing",
         "occupancy_non_missing"]
    ].head(20)
)

# 1. Settings
# ------------------------------------------------------------

P2_OUT = output_dir
P2_OUT.mkdir(exist_ok=True, parents=True)

P2_START = "2020-01-01"
P2_END = "2024-08-14"

P2_MAX_LAG = 6
P2_BASIS_DIM = 3
P2_TREND_DF = 8
P2_AR_ORDER = 1
P2_MIN_WEEKS = 80

# Keep False until admissions reporting intervals have been checked.
P2_ADMISSIONS_ENDPOINTS_VERIFIED = False

required = [
    "location",
    "iso_code",
    "date",
    "new_cases_per_million",
    "weekly_hosp_admissions_per_million",
    "hosp_patients_per_million",
]

numeric_columns = [
    "new_cases_per_million",
    "weekly_hosp_admissions_per_million",
    "hosp_patients_per_million",
]

keys = ["location", "date"]
value_columns = ["iso_code"] + numeric_columns

# ------------------------------------------------------------
# 2. Validate notebook inputs
# ------------------------------------------------------------

if "df_global" not in globals():
    raise RuntimeError("Run your df_global data-loading cell first.")

missing = set(required) - set(df_global.columns)

if missing:
    raise ValueError(f"Missing raw-data columns: {sorted(missing)}")

if (
    "tier1" not in globals()
    or not isinstance(tier1, pd.DataFrame)
    or "location" not in tier1.columns
):
    raise RuntimeError("Run your Tier 1 definition cell first.")

p2_candidates = sorted(
    tier1["location"].dropna().astype(str).unique()
)

if not p2_candidates:
    raise ValueError("Your Tier 1 country list is empty.")

# ------------------------------------------------------------
# 3. Select the study sample BEFORE checking duplicates
# ------------------------------------------------------------

p2_raw = df_global[required].copy()

p2_raw["date"] = pd.to_datetime(
    p2_raw["date"], errors="raise"
).dt.normalize()

real_country = (
    p2_raw["iso_code"].notna()
    & ~p2_raw["iso_code"].astype("string").str.startswith(
        "OWID_", na=False
    )
)

p2_raw = p2_raw.loc[
    real_country
    & p2_raw["location"].isin(p2_candidates)
    & p2_raw["date"].between(P2_START, P2_END)
].copy()

if p2_raw.empty:
    raise ValueError("No records remain after sample selection.")

if p2_raw[keys].isna().any().any():
    raise ValueError("Missing country names or dates in selected records.")

# Require valid numeric values; do not silently convert bad text to NaN.
for column in numeric_columns:
    p2_raw[column] = pd.to_numeric(
        p2_raw[column], errors="raise"
    )

if np.isinf(p2_raw[numeric_columns].to_numpy(dtype=float)).any():
    raise ValueError("Infinite indicator values found in selected records.")

print("Candidate Tier 1 countries:", len(p2_candidates))
print("Selected rows before duplicate handling:", len(p2_raw))

# ------------------------------------------------------------
# 4. Remove completely identical copies
# ------------------------------------------------------------

rows_before = len(p2_raw)
p2_raw = p2_raw.drop_duplicates().copy()
identical_removed = rows_before - len(p2_raw)

print("Identical duplicate rows removed:", identical_removed)

# ------------------------------------------------------------
# 5. Inspect remaining country-date duplicates
#
# Compatible:
#   16.07 and NaN -> retain 16.07
#   0.00 and NaN  -> retain 0.00
#   NaN and NaN   -> retain NaN
#
# Conflicting:
#   16.07 and 18.20 -> stop for inspection
# ------------------------------------------------------------

duplicate_mask = p2_raw.duplicated(keys, keep=False)

p2_duplicate_audit = pd.DataFrame(
    columns=keys + ["n_rows", "conflicting_columns"]
)

if duplicate_mask.any():
    duplicate_rows = p2_raw.loc[duplicate_mask].copy()
    duplicate_groups = duplicate_rows.groupby(keys, sort=True)

    # Count distinct OBSERVED values, excluding missing values.
    distinct_values = duplicate_groups[value_columns].nunique(
        dropna=True
    )

    conflict_flags = distinct_values.gt(1)

    p2_duplicate_audit = duplicate_groups.size().rename(
        "n_rows"
    ).to_frame()

    p2_duplicate_audit["conflicting_columns"] = conflict_flags.apply(
        lambda row: ", ".join(row.index[row].tolist()),
        axis=1
    )

    p2_duplicate_audit = p2_duplicate_audit.reset_index()

    p2_duplicate_audit.to_csv(
        P2_OUT / "duplicate_resolution_audit.csv",
        index=False
    )

    conflicting_keys = (
        conflict_flags.loc[conflict_flags.any(axis=1)]
        .reset_index()[keys]
    )

    if not conflicting_keys.empty:
        conflicting_records = (
            duplicate_rows
            .merge(conflicting_keys, on=keys, how="inner")
            .sort_values(keys)
        )

        conflicting_records.to_csv(
            P2_OUT / "conflicting_country_date_records.csv",
            index=False
        )

        print(
            "Groups with contradictory observed values:",
            len(conflicting_keys)
        )
        display(conflicting_records.head(40))

        raise ValueError(
            "Different observed values remain for the same country-date "
            "and field. Inspect conflicting_country_date_records.csv. "
            "These records have not been averaged or arbitrarily selected."
        )

    # Safe because each field has at most ONE distinct observed value.
    # first() selects the first non-missing value independently per field.
    rows_before_merge = len(p2_raw)

    p2_raw = (
        p2_raw.groupby(keys, as_index=False, sort=True)[value_columns]
        .first()
    )

    print(
        "Compatible duplicate country-date groups combined:",
        len(p2_duplicate_audit)
    )
    print(
        "Redundant rows removed by combination:",
        rows_before_merge - len(p2_raw)
    )

else:
    p2_duplicate_audit.to_csv(
        P2_OUT / "duplicate_resolution_audit.csv",
        index=False
    )
    print("No remaining duplicate country-date groups.")

# ------------------------------------------------------------
# 6. Final validation
# ------------------------------------------------------------

p2_raw = (
    p2_raw[required]
    .sort_values(keys)
    .reset_index(drop=True)
)

if p2_raw.duplicated(keys).any():
    raise RuntimeError("Country-date uniqueness check failed.")

# A country name should identify one ISO code in this sample.
iso_counts = p2_raw.groupby("location")["iso_code"].nunique()

if (iso_counts > 1).any():
    raise ValueError(
        "Country names associated with multiple ISO codes: "
        f"{iso_counts[iso_counts > 1].index.tolist()}"
    )

available_countries = sorted(p2_raw["location"].unique())
absent_countries = sorted(set(p2_candidates) - set(available_countries))

print("\nCountry-date uniqueness confirmed.")
print("Countries with selected records:", len(available_countries))
print("Final selected rows:", len(p2_raw))
print("Observed date range:", p2_raw["date"].min(), "to", p2_raw["date"].max())

if absent_countries:
    print("Tier 1 countries without selected records:", absent_countries)

display(p2_raw.head())

# ============================================================
# NEW ANALYSIS 2: Weekly alignment and coverage audit
# Replaces the entire previous Chunk 2
# ============================================================

p2_weekly_parts = []
p2_alignment_rows = []

if "p2_raw" not in globals():
    raise RuntimeError("Run the updated Chunk 1 first.")

if p2_raw.duplicated(["location", "date"]).any():
    raise ValueError(
        "Duplicate country-date records remain. "
        "Run the updated Chunk 1 before this chunk."
    )

for country, g in p2_raw.groupby("location"):

    # Preserve every calendar day, including missing dates.
    daily = (
        g.set_index("date")
        .sort_index()
        .reindex(pd.date_range(P2_START, P2_END, freq="D"))
    )

    cases = pd.to_numeric(
        daily["new_cases_per_million"],
        errors="raise"
    )

    # Negative corrections are excluded rather than interpreted
    # as negative infections.
    negative_cases = int((cases < 0).sum())
    cases = cases.mask(cases < 0)

    occupancy = pd.to_numeric(
        daily["hosp_patients_per_million"],
        errors="raise"
    )
    negative_occupancy = int((occupancy < 0).sum())
    occupancy = occupancy.mask(occupancy < 0)

    admissions = pd.to_numeric(
        daily["weekly_hosp_admissions_per_million"],
        errors="raise"
    )
    negative_admissions = int((admissions < 0).sum())
    admissions = admissions.mask(admissions < 0)

    # Monday–Sunday intervals, labelled by the Sunday endpoint.
    weekly_index = cases.resample(
        "W-SUN", closed="right", label="right"
    ).sum(min_count=7).index

    w = pd.DataFrame(index=weekly_index)

    # Require all seven daily case values.
    w["cases"] = cases.resample(
        "W-SUN", closed="right", label="right"
    ).sum(min_count=7)

    # Require all seven daily occupancy values.
    occupancy_count = occupancy.resample(
        "W-SUN", closed="right", label="right"
    ).count()

    w["occupancy"] = occupancy.resample(
        "W-SUN", closed="right", label="right"
    ).mean().where(occupancy_count == 7)

    # OWID defines this admissions indicator as the observation
    # date plus the preceding six days.
    #
    # Use the EXACT Sunday observation for Monday–Sunday.
    # Do not sum rolling admissions totals, interpolate,
    # forward fill, or substitute another weekday.
    w["admissions"] = admissions.reindex(w.index)

    w["location"] = country
    w.index.name = "week_end"

    p2_weekly_parts.append(w.reset_index())

    observed_admission_dates = daily.index[admissions.notna()]

    weekday_counts = (
        pd.Series(observed_admission_dates.dayofweek, dtype="int64")
        .value_counts()
        .sort_index()
        .to_dict()
    )

    paired_weeks = int(
        w[["cases", "admissions", "occupancy"]]
        .notna()
        .all(axis=1)
        .sum()
    )

    p2_alignment_rows.append({
        "location": country,
        "negative_case_days_removed": negative_cases,
        "negative_admissions_days_removed": negative_admissions,
        "negative_occupancy_days_removed": negative_occupancy,
        "daily_admissions_available": int(admissions.notna().sum()),
        "sunday_admissions_available": int(w["admissions"].notna().sum()),
        "complete_cases_weeks": int(w["cases"].notna().sum()),
        "complete_occupancy_weeks": int(w["occupancy"].notna().sum()),
        "paired_weeks_before_lags": paired_weeks,
        "below_minimum_before_lags": paired_weeks < P2_MIN_WEEKS,
        "admissions_weekday_counts_0_Monday": str(weekday_counts),
    })

if not p2_weekly_parts:
    raise ValueError("No countries are available for weekly aggregation.")

p2_weekly = (
    pd.concat(p2_weekly_parts, ignore_index=True)
    .sort_values(["location", "week_end"])
    .reset_index(drop=True)
)

p2_alignment = (
    pd.DataFrame(p2_alignment_rows)
    .sort_values("location")
    .reset_index(drop=True)
)

if p2_weekly.duplicated(["location", "week_end"]).any():
    raise RuntimeError("Duplicate country-week records after aggregation.")

# Dataset-level definition checked against OWID's codebook.
# This does not establish identical country reporting practices.
P2_ADMISSIONS_ENDPOINTS_VERIFIED = True

p2_alignment_metadata = {
    "admissions_definition_source": (
        "https://raw.githubusercontent.com/owid/covid-19-data/"
        "f0d53320b19703cf565940d8269181a8423d7032/public/data/owid-covid-codebook.csv"
    ),
    "admissions_interval": "Observation date and preceding six days",
    "weekly_endpoint": "Sunday",
    "cases_aggregation": "Sum; seven observed daily values required",
    "occupancy_aggregation": "Mean; seven observed daily values required",
    "admissions_aggregation": "Exact Sunday observation",
    "missing_value_filling": "None",
    "verification_scope": (
        "OWID dataset-level definition; country reporting "
        "comparability is not established by this check"
    ),
}

# Save audit, weekly data and alignment definitions.
p2_alignment.to_csv(
    P2_OUT / "weekly_alignment_audit.csv",
    index=False
)

p2_weekly.to_csv(
    P2_OUT / "weekly_aligned_data.csv",
    index=False
)

(P2_OUT / "weekly_alignment_metadata.json").write_text(
    json.dumps(p2_alignment_metadata, indent=2),
    encoding="utf-8"
)

display(p2_alignment)

print(
    "\nWeekly alignment completed using OWID's documented "
    "seven-day admissions interval."
)

p2_low_coverage = p2_alignment.loc[
    p2_alignment["below_minimum_before_lags"],
    ["location", "paired_weeks_before_lags"]
]

if not p2_low_coverage.empty:
    print(
        "\nCountries below the minimum of "
        f"{P2_MIN_WEEKS} paired weeks BEFORE lag construction:"
    )
    display(p2_low_coverage)

    print(
        "Chunk 3 will exclude these countries and record the reason. "
        "Additional countries may fail after requiring complete lag histories."
    )

print("\nContinue with Chunk 3.")
reference_dir = Path(__file__).resolve().parent
checks = {}
for filename in ['weekly_aligned_data.csv', 'weekly_alignment_audit.csv']:
    actual = pd.read_csv(output_dir / filename)
    expected = pd.read_csv(reference_dir / ('archived_' + filename))
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    checks[filename] = {'rows': len(actual), 'columns': list(actual.columns), 'matches': True}
(output_dir / 'verification.json').write_text(json.dumps(checks, indent=2))
print('PASS: archived weekly values, missingness, rows and alignment audit reproduced.')
