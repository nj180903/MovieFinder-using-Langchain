import pandas as pd
import kagglehub

def load_dataset():
    path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
    df = pd.read_csv(f"{path}/imdb_top_1000.csv")

    df = df.dropna(subset=["Series_Title", "Overview"])
    df['combined'] = df['Series_Title'] + ". " + df['Overview']
    print(df["combined"][0])
    print(df['combined'][100])
    print(df.head())
    return df


# load_dataset()