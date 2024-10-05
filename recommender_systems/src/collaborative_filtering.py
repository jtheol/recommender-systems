from typing import Tuple
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


class MovieLensData:
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        users = self.users[item]
        movies = self.movies[item]
        ratings = self.ratings[item]

        return {
            "users": torch.tensor(users, dtype=torch.long),
            "movies": torch.tensor(movies, dtype=torch.long),
            "ratings": torch.tensor(ratings, dtype=torch.long),
        }


class MovieRecommender(nn.Module):
    def __init__(self, n_users, n_movies, emb_size=32):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, emb_size)
        self.item_emb = nn.Embedding(n_movies, emb_size)
        self.out = nn.Linear(64, 1)

    def forward(self, users, movies):
        user_embeds = self.user_emb(users)
        movie_embeds = self.item_emb(movies)
        output = torch.cat([user_embeds, movie_embeds], dim=1)

        output = self.out(output)

        return output


def build_dataset(data: DataFrame, seed: int) -> Tuple[MovieLensData, MovieLensData]:
    le_user = LabelEncoder()
    le_movie = LabelEncoder()

    data["userId"] = le_user.fit_transform(data["userId"].values)
    data["movieId"] = le_movie.fit_transform(data["movieId"].values)

    train, test = train_test_split(
        data, test_size=0.2, random_state=seed, stratify=data["rating"].values
    )

    train_data = MovieLensData(
        users=train["userId"].values,
        movies=train["movieId"].values,
        ratings=train["rating"].values,
    )

    test_data = MovieLensData(
        users=test["userId"].values,
        movies=test["movieId"].values,
        ratings=test["rating"].values,
    )

    return train_data, test_data


def build_dataloaders(
    train_data: MovieLensData, test_data: MovieLensData, batch_size: int = 4
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return train_loader, test_loader


def train_model(
    model,
    train_loader,
    epochs: int = 1,
    lr: float = 0.001,
    step_size: int = 3,
    gamma: float = 0.7,
) -> list:

    loss_fn = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = StepLR(optim, step_size=step_size, gamma=gamma)

    loss_val = 0.0
    loss_hist = []

    model.train()
    for epoch in range(epochs):
        for _, train_data in enumerate(train_loader):

            users_data = train_data["users"].to(device)
            movies_data = train_data["movies"].to(device)
            ratings_data = train_data["ratings"].to(device)

            output = model(users_data, movies_data)
            rating = ratings_data.view(4, -1).to(torch.float32).to(device)

            optim.zero_grad()
            loss = loss_fn(output, rating)
            loss.backward()
            optim.step()

            loss_val += loss.item()

            avg_loss = loss_val / len(train_data["users"])

            print(f"Epoch: {epoch}, loss is {avg_loss}")

            loss_hist.append(avg_loss)
            loss_val = 0.0

        scheduler.step()

    return loss_hist


if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("mps")

    print(f"Using device: {device}")

    movie_lens_data = pd.read_csv("../data/ml-latest-small/ratings.csv")
    train, test = build_dataset(data=movie_lens_data, seed=1052024)
    train_loader, test_loader = build_dataloaders(train_data=train, test_data=test)

    model = MovieRecommender(
        n_users=len(movie_lens_data["userId"].unique()),
        n_movies=len(movie_lens_data["movieId"].unique()),
        emb_size=32,
    ).to(device)

    loss_hist = train_model(model, train_loader, epochs=3)
