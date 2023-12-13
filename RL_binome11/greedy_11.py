#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This program is an improvement of "greedy_1".
    A problem of the previous version is that once the route is found, it is fixed once and for all.
    If we introduce an opponent in the maze, this will definitely not work out.
    In this version, we make the approach responsive by performing a greedy search at every turn.
    This will lead to the same route if there is no opponent, and will take disappearance of cheese into account if there is one.
    To illustrate, we introduce a second player in the form of "random_5".
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Import PyRat
from pyrat import *

# External imports 
import numpy

# Previously developed functions
from utils import bfs, find_route, locations_to_action

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def give_score ( graph:          Union[numpy.ndarray, Dict[int, Dict[int, int]]],
                 current_vertex: int,
                 targets:        List[int]
               ) ->              Tuple[List[float], Dict[int, Union[None, int]]]:

    """
        Function that associates a score to each target.
        In:
            * graph:          Graph containing the vertices.
            * current_vertex: Current location of the player in the maze.
            
        Out:
            * scores:        Scores given to the targets.
            * routing_table: Routing table obtained from the current vertex.
    """
    
    # We compute distances from the current vertex to targets
    distances_to_explored_vertices, routing_table = bfs(current_vertex, graph)
    
    # Scores are inversely proportional to that distance
    scores = [1.0 / distances_to_explored_vertices[target] for target in targets]
    return scores, routing_table

#####################################################################################################################################################
######################################################### EXECUTED AT EACH TURN OF THE GAME #########################################################
#####################################################################################################################################################

def turn ( maze:             Union[numpy.ndarray, Dict[int, Dict[int, int]]],
           maze_width:       int,
           maze_height:      int,
           name:             str,
           teams:            Dict[str, List[str]],
           player_locations: Dict[str, int],
           player_scores:    Dict[str, float],
           player_muds:      Dict[str, Dict[str, Union[None, int]]],
           cheese:           List[int],
           possible_actions: List[str],
           memory:           threading.local
         ) ->                str:

    """
        This function is called at every turn of the game and should return an action within the set of possible actions.
        You can access the memory you stored during the preprocessing function by doing memory.my_key.
        You can also update the existing memory with new information, or create new entries as memory.my_key = my_value.
        In:
            * maze:             Map of the maze, as data type described by PyRat's "maze_representation" option.
            * maze_width:       Width of the maze in number of cells.
            * maze_height:      Height of the maze in number of cells.
            * name:             Name of the player controlled by this function.
            * teams:            Recap of the teams of players.
            * player_locations: Locations for all players in the game.
            * player_scores:    Scores for all players in the game.
            * player_muds:      Indicates which player is currently crossing mud.
            * cheese:           List of available pieces of cheese in the maze.
            * possible_actions: List of possible actions.
            * memory:           Local memory to share information between preprocessing, turn and postprocessing.
        Out:
            * action: One of the possible actions, as given in possible_actions.
    """
    
    # We find the next cheese to get using the greedy method
    scores, routing_table = give_score(maze, player_locations[name], cheese)
    best_target = cheese[numpy.argmax(scores)]
    route = find_route(routing_table, player_locations[name], best_target)
    
    # We move once in that direction
    action = locations_to_action(route[0], route[1], maze_width)
    return action

#####################################################################################################################################################
#####################################################################################################################################################
