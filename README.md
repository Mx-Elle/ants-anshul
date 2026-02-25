[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/L89OX_6N)
# Ants AI Competition

## Movement

Every tick, each ant may move one space north, south, east, or west. If you try to make an ant move illegally, it will stay still.

If after moving, multiple ants would occupy the same space, all of those ants die. An ant can safely into a space that is currently occupied by another ant as long as that ant moves out of the way in the same tick.

Each tick, both players will queue up all their intended moves and then every ant will move simultaneously.

## Combat

After all ants have moved and ants that collided have been removed, the ants will fight. At the start of the game you will be told what the combat radius is.
Each ant deals one damage split evenly between all enemies within their combat radius. After all ants have dealt their damage, any ants that received at least 1 damage are killed. Damage does not carry over between ticks.

Effectively, this means that a big group will kill a smaller group of enemies without dying themselves, and equal groups will receive equal casualties.

## Harvesting

After combat, ants attempt to harvest food. At the start of the game you will be told what the harvest radius is. Each piece of food will check what ants are within that radius of it. If all the ants in harvest radius of the food are the same team, that team gets the food and it is added to their total food. If there are ants from both teams within the harvest radius, the food is destroyed in the scuffle and nobody gets it.

## Spawning Ants

At the start of each tick (before your AI assigns moves) each anthill will consume 1 food and spawn 1 ant. (So if you have >=3 food and 3 anthills, all three anthills will spawn a new ant.) A hill cannot spawn an ant if there is already a friendly ant on top of it. (So if you want a hill to not spawn ants, you can sit an ant on top of it to stop it.) Spawning an ant consumes one food. If there is not enough food to spawn ants at all of your hills, hills are prioritized by how long it's been since the hill last spawned an ant.

## Destroying hills

If you manage to move an ant onto your opponent's hill, that hill is permanently destroyed.

## Spawning food

Food spawns symmetrically onto the board. At the beginning of the game, all possible food spawn locations are shuffled and then food spawns on those spaces in order until every space has spawned food. Then the locations get reshuffled and go again. If a space already has food when it would spawn, nothing happens.

## Winning the game

If one player has no hills left, they instantly lose. Otherwise, if the tick limit is reached, the player with the most remaining hills wins. If at this point, both players have the same number of hills, then the players are each given a score equal to (food stored) + (ants on the board), and the player with the higher score wins. If this is still a tie, then the game is declared a draw.

## Vision

Every tick you will be told the locations of all of your ants and anthills, as well as the locations of food, enemy ants, and enemy hills within your vision radius. (Although since the hills are symmetrical, you can figure out where the opposing hills are based on where your hills are.) Walls do not block vision.
