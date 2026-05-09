import pandas as pd
import numpy as np
from pathlib import Path


AGE_MAP = {1: "<18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
OCCUPATION_MAP = {
    0: "Other", 1: "Academic/Educator", 2: "Artist", 3: "Clerical/Admin",
    4: "College/Grad student", 5: "Customer service", 6: "Doctor/Health care",
    7: "Executive/Managerial", 8: "Farmer", 9: "Homemaker", 10: "K-12 student",
    11: "Lawyer", 12: "Programmer", 13: "Retired", 14: "Sales/Marketing",
    15: "Scientist", 16: "Self-employed", 17: "Technician/Engineer",
    18: "Tradesman/Craftsman", 19: "Unemployed", 20: "Writer",
}


class DataLoader:
    def load_movies(self, path: str, ratings_df: pd.DataFrame | None = None) -> pd.DataFrame:
        df = pd.read_csv(
            path,
            sep="::",
            engine="python",
            encoding="latin-1",
            names=["movieId", "title", "genres"],
        )
        df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype(float)
        df["genres_str"] = df["genres"].str.replace("|", " ", regex=False)
        df["genres_list"] = df["genres"].str.split("|")
        if ratings_df is not None:
            stats = ratings_df.groupby("movieId")["rating"].agg(
                avg_rating="mean", n_ratings="count"
            ).reset_index()
            df = df.merge(stats, on="movieId", how="left")
            df["avg_rating"] = df["avg_rating"].fillna(0.0)
            df["n_ratings"] = df["n_ratings"].fillna(0).astype(int)
        return df

    def load_ratings(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(
            path,
            sep="::",
            engine="python",
            encoding="latin-1",
            names=["userId", "movieId", "rating", "timestamp"],
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    def load_users(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(
            path,
            sep="::",
            engine="python",
            encoding="latin-1",
            names=["userId", "gender", "age", "occupation", "zip"],
        )
        df["age_label"] = df["age"].map(AGE_MAP)
        df["occupation_label"] = df["occupation"].map(OCCUPATION_MAP)
        return df
