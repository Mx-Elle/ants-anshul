from copy import deepcopy
from dataclasses import dataclass
import numbers
import sys
from time import monotonic
from typing import Protocol, Type

import numpy as np
import numpy.typing as npt
import pygame
import pygame.locals
from tqdm import trange
from board import Board, Entity, cells_within_distance, generate_board
from dataclasses import dataclass

from random_player import RandomBot

AntMove = tuple[tuple[int, int], tuple[int, int]]


class Player(Protocol):
    """
    Your player must match this protocol. (It must have each of these functions with the same signatures)
    The constructor provides your player object with the information that won't change during the game.
    You must have a name property that returns a string

    Your move function takes as input the things you can see as a set of tuples containing a coordinate and an Enum
    telling you what kind of thing is at that location. This will always include all of your ants and hills. Your ants
    and hills can see enemy ants, enemy hills, and food in a certain radius of them.

    You will push all of your moves into the provided move_queue in the form (start_loc, end_loc) where both locations
    are given as tuple[int, int]. When you run out of time for the turn, the function will be terminated and whatever moves
    you managed to push in time will be executed.
    """

    def __init__(
        self,
        walls: npt.NDArray[np.int_],
        harvest_radius: int,
        vision_radius: int,
        battle_radius: int,
        max_turns: int,
        time_per_turn: float,
    ) -> None: ...

    @property
    def name(self) -> str: ...

    def move_ants(
        self,
        vision: set[tuple[tuple[int, int], Entity]],
        stored_food: int,
    ) -> set[AntMove]: ...


@dataclass
class GameSpecification:

    board: Board
    harvest_radius: int = 1
    vision_radius: int = 8
    battle_radius: int = 3
    max_turns: int = 1000
    time_per_turn: float = 0.3


def play_game(
    spec: GameSpecification,
    p1_type: Type[Player],
    p2_type: Type[Player],
    visualize: bool = True,
):
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))

    p1, p2 = p1_type(
        deepcopy(spec.board.walls),
        spec.harvest_radius,
        spec.vision_radius,
        spec.battle_radius,
        spec.max_turns,
        spec.time_per_turn,
    ), p2_type(
        deepcopy(spec.board.walls),
        spec.harvest_radius,
        spec.vision_radius,
        spec.battle_radius,
        spec.max_turns,
        spec.time_per_turn,
    )
    board = deepcopy(spec.board)
    p1_hills = {h: 0 for h in zip(*np.where(board.hills == 1))}
    p2_hills = {h: 0 for h in zip(*np.where(board.hills == 2))}
    food = {1: len(p1_hills), 2: len(p2_hills)}
    for _ in trange(spec.max_turns):
        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                pygame.quit()
                sys.exit()

        spawn_ants(board, food, p1_hills, p2_hills)
        try:
            p1_moves, p2_moves = run_players(spec, p1, p2, board, food)
        except TimeoutError as e:
            if "p1" in str(e):
                print(f"Blue {p2.name} wins due to opponent timing out.")
            if "p2" in str(e):
                print(f"Red {p1.name} wins due to opponent timing out.")
        move_ants(board, p1_moves, p2_moves)
        combat(board, spec.battle_radius)
        flatten_hills(board)
        harvest(board, spec.harvest_radius, food)
        board.spawn_food()

        if visualize:
            screen.blit(board.render(screen.get_width(), screen.get_height()), (0, 0))
            pygame.display.flip()

        p1_hill_count = len(set(zip(*np.where(board.hills == 1))))
        p2_hill_count = len(set(zip(*np.where(board.hills == 2))))
        if p1_hill_count == 0:
            print(f"Blue {p2.name} wins by destroying all opposing hills")
            return
        if p2_hill_count == 0:
            print(f"Red {p1.name} wins by destroying all opposing hills")
            return
    p1_hill_count = len(set(zip(*np.where(board.hills == 1))))
    p2_hill_count = len(set(zip(*np.where(board.hills == 2))))
    if p1_hill_count > p2_hill_count:
        print(
            f"Red {p1.name} wins by having more leftover hills. ({p1_hill_count} to {p2_hill_count})"
        )
    elif p2_hill_count > p1_hill_count:
        print(
            f"Blue {p2.name} wins by having more leftover hills. ({p2_hill_count} to {p1_hill_count})"
        )
    else:
        p1_score = food[1] + len(set(zip(*np.where(board.ants == 1))))
        p2_score = food[2] + len(set(zip(*np.where(board.ants == 2))))
        if p1_score > p2_score:
            print(f"Red {p1.name} wins on score. ({p1_score} to {p2_score})")
        elif p2_score > p1_score:
            print(f"Blue {p2.name} wins on score. ({p2_score} to {p1_score})")
        else:
            print(f"Drawn game!")


def validate(move: AntMove) -> bool:
    try:
        if len(move) != 2:
            return False
        for elem in move:
            if len(elem) != 2:
                return False
            if not (
                isinstance(elem[0], numbers.Integral)
                and isinstance(elem[1], numbers.Integral)
            ):
                return False
        return True
    except:
        return False


def run_players(
    spec: GameSpecification, p1: Player, p2: Player, board: Board, food: dict[int, int]
) -> tuple[set[AntMove], set[AntMove]]:
    start = monotonic()
    try:
        p1_moves = p1.move_ants(board.get_vision(1, spec.vision_radius), food[1])
    except:
        p1_moves = set()
    p1_time = monotonic() - start
    if p1_time > spec.time_per_turn:
        raise TimeoutError("p1 timeout")
    start = monotonic()
    try:
        p2_moves = p2.move_ants(board.get_vision(2, spec.vision_radius), food[2])
    except:
        p2_moves = set()
    p2_time = monotonic() - start
    if p2_time > spec.time_per_turn:
        raise TimeoutError("p2 timeout")
    p1_moves = {
        (board.wrap(move[0]), board.wrap(move[1]))
        for move in p1_moves
        if validate(move)
    }
    p2_moves = {
        (board.wrap(move[0]), board.wrap(move[1]))
        for move in p2_moves
        if validate(move)
    }
    return p1_moves, p2_moves


def move_ants(board: Board, p1_moves: set[AntMove], p2_moves: set[AntMove]) -> None:
    p1_actions = {
        start: end
        for start, end in p1_moves
        if start != end
        and start in set(zip(*np.where(board.ants == 1)))
        and not board.walls[end]
        and end in cells_within_distance(1, start, board.shape)
    }
    p2_actions = {
        start: end
        for start, end in p2_moves
        if start != end
        and start in set(zip(*np.where(board.ants == 2)))
        and not board.walls[end]
        and end in cells_within_distance(1, start, board.shape)
    }
    p1_origins, p1_destinations = p1_actions.keys(), list(p1_actions.values())
    p2_origins, p2_destinations = p2_actions.keys(), list(p2_actions.values())
    for origin in p1_origins | p2_origins:
        board.ants[origin] = 0
    all_destinations = p1_destinations + p2_destinations
    entrants = {c: all_destinations.count(c) for c in all_destinations}
    for destination in p1_destinations:
        if entrants[destination] == 1:
            board.ants[destination] = 1
    for destination in p2_destinations:
        if entrants[destination] == 1:
            board.ants[destination] = 2


def spawn_ants(
    board: Board,
    food: dict[int, int],
    p1_hills: dict[tuple[int, int], int],
    p2_hills: dict[tuple[int, int], int],
) -> None:
    for hill in p1_hills:
        p1_hills[hill] += 1
    for hill in p2_hills:
        p2_hills[hill] += 1
    eligible_p1_hills = {
        h for h in p1_hills if board.hills[h] == 1 and not board.ants[h]
    }
    eligible_p2_hills = {
        h for h in p2_hills if board.hills[h] == 2 and not board.ants[h]
    }
    while eligible_p1_hills and food[1]:
        hill = max(eligible_p1_hills, key=lambda x: p1_hills[x])
        food[1] -= 1
        p1_hills[hill] = 0
        board.ants[hill] = 1
        eligible_p1_hills.remove(hill)
    while eligible_p2_hills and food[2]:
        hill = max(eligible_p2_hills, key=lambda x: p2_hills[x])
        food[2] -= 1
        p1_hills[hill] = 0
        board.ants[hill] = 2
        eligible_p2_hills.remove(hill)


def combat(board: Board, battle_radius: int) -> None:
    p1_ants = set(zip(*np.where(board.ants == 1)))
    p2_ants = set(zip(*np.where(board.ants == 2)))
    p1_ant_damage = {ant: 0.0 for ant in p1_ants}
    p2_ant_damage = {ant: 0.0 for ant in p2_ants}
    for ant in p1_ants:
        enemies = cells_within_distance(battle_radius, ant, board.shape) & p2_ants
        if not enemies:
            continue
        damage = 1 / len(enemies)
        for enemy in enemies:
            p2_ant_damage[enemy] += damage
    for ant in p2_ants:
        enemies = cells_within_distance(battle_radius, ant, board.shape) & p1_ants
        if not enemies:
            continue
        damage = 1 / len(enemies)
        for enemy in enemies:
            p1_ant_damage[enemy] += damage
    for ant in {a for a in p1_ant_damage if p1_ant_damage[a] >= 1} | {
        a for a in p2_ant_damage if p2_ant_damage[a] >= 1
    }:
        board.ants[ant] = 0


def flatten_hills(board: Board) -> None:
    for hill in zip(*np.where(board.hills)):
        if board.ants[hill] != 0 and board.ants[hill] != board.hills[hill]:
            board.hills[hill] = 0


def harvest(board: Board, collect_radius: int, food: dict[int, int]) -> None:
    all_ants: set[tuple[int, int]] = set(zip(*np.where(board.ants != 0)))
    all_food = set(zip(*np.where(board.food)))
    for f in all_food:
        ants = cells_within_distance(collect_radius, f, board.shape) & all_ants
        ant_colors = {board.ants[ant] for ant in ants}
        if len(ant_colors) >= 1:
            board.food[f] = 0
        if len(ant_colors) == 1:
            food[ant_colors.pop()] += 1


def main():
    b = generate_board(60, 60, hills_per_player=2)
    spec = GameSpecification(b)
    play_game(spec, RandomBot, RandomBot)


if __name__ == "__main__":
    main()
