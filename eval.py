import os

import torch

from agent import Player
from ant_game import GameSpecification, play_game
from board import generate_board
from random_player import RandomBot


def main():
    board = generate_board(60, 60, hills_per_player=3)
    spec = GameSpecification(
        board=board,
        harvest_radius=1,
        vision_radius=8,
        battle_radius=3,
        max_turns=1000,
        time_per_turn=0.3,
    )

    latest_path = "models/latest.pt"
    default_path = "ant_model.pt"
    model_to_use = latest_path if os.path.exists(latest_path) else default_path

    steps = 0
    if os.path.exists(model_to_use):
        checkpoint = torch.load(model_to_use, map_location="cpu", weights_only=False)
        steps = checkpoint.get("global_step", 0)

    print(f"Loading model: {model_to_use} (trained for {steps:,} steps)")

    def p1_factory(*args, **kwargs):
        p = Player(*args, **kwargs, model_path=model_to_use)
        p.name = f"AntNet ({steps:,} steps)"
        return p

    play_game(spec, p1_factory, RandomBot, visualize=True)


if __name__ == "__main__":
    main()
