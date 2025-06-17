from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_excel(path: str | Path, sheet: str | None = None) -> pd.DataFrame:
    """Load an Excel sheet into a DataFrame."""
    return pd.read_excel(path, sheet_name=sheet)


def ade_report(df: pd.DataFrame) -> None:
    """Print basic info, missing value count and descriptive stats."""
    print("=== INFO ===")
    print(df.info())
    print("\n=== MISSING PER COLUMN ===")
    print(df.isna().sum())
    print("\n=== NUMERIC SUMMARY ===")
    print(df.describe().round(2))


def ade_visuals(df: pd.DataFrame, out_dir: str = "ade_plots") -> None:
    """Generate histograms for numeric columns and a correlation heatmap."""
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        sns.histplot(df[col].dropna())
        plt.title(col)
        plt.savefig(out / f"{col}_hist.png")
        plt.clf()
    if numeric.shape[1] > 1:
        sns.heatmap(numeric.corr(), annot=True, fmt=".2f")
        plt.title("Correlation heatmap")
        plt.savefig(out / "correlation_heatmap.png")
        plt.clf()


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Simple ADE report")
    p.add_argument("xlsx", help="Excel file")
    p.add_argument("--sheet", default=None, help="Sheet name")
    p.add_argument("--plots", action="store_true", help="Generate plot images")
    p.add_argument("--out", default="ade_plots", help="Output directory for plots")
    args = p.parse_args()

    df = load_excel(args.xlsx, sheet=args.sheet)
    ade_report(df)
    if args.plots:
        ade_visuals(df, args.out)

if __name__ == "__main__":
    main()
