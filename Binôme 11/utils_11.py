#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This file compiles useful functions to work with PyRat.
    It includes functions to:
        * Manipulate a graph.
        * Perform a traversal.
        * Help in the game.
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Import PyRat
from pyrat import *

#####################################################################################################################################################
############################################################ GRAPH MANIPULATION FUNCTIONS ###########################################################
#####################################################################################################################################################

def get_vertices ( graph: Union[numpy.ndarray, Dict[int, Dict[int, int]]]
                 ) ->     List[int]:

    """
        Function to return the list of all vertices in a graph, except those with no neighbors.
        Here we propose an implementation for all types handled by the PyRat game.
        In:
            * graph: Graph on which to get the list of vertices.
        Out:
            * vertices: List of vertices in the graph.
    """
    
    # If "maze_representation" option is set to "dictionary"
    if isinstance(graph, dict):
        vertices = list(graph.keys())
    
    # If "maze_representation" option is set to "matrix"
    elif isinstance(graph, numpy.ndarray):
        vertices = list(graph.sum(axis=0).nonzero()[0])
    
    # Unhandled data type
    else:
        raise Exception("Unhandled graph type", type(graph))
    
    # Done
    return vertices

#####################################################################################################################################################

def get_neighbors ( vertex: int,
                    graph:  Union[numpy.ndarray, Dict[int, Dict[int, int]]]
                  ) ->      List[int]:

    """
        Function to return the list of neighbors of a given vertex.
        Here we propose an implementation for all types handled by the PyRat game.
        The function assumes that the vertex exists in the maze.
        It can be checked using for instance `assert vertex in get_vertices(graph)` but this takes time.
        In:
            * vertex: Vertex for which to compute the neighborhood.
            * graph:  Graph on which to get the neighborhood of the vertex.
        Out:
            * neighbors: List of vertices that are adjacent to the vertex in the graph.
    """
    
    # If "maze_representation" option is set to "dictionary"
    if isinstance(graph, dict):
        neighbors = list(graph[vertex].keys())

    # If "maze_representation" option is set to "matrix"
    elif isinstance(graph, numpy.ndarray):
        neighbors = graph[vertex].nonzero()[0].tolist()
    
    # Unhandled data type
    else:
        raise Exception("Unhandled graph type", type(graph))
    
    # Done
    return neighbors
    
#####################################################################################################################################################

def get_weight ( source: int,
                 target: int,
                 graph:  Union[numpy.ndarray, Dict[int, Dict[int, int]]]
               ) ->      List[int]:

    """
        Function to return the weight of the edge in the graph from the source to the target.
        Here we propose an implementation for all types handled by the PyRat game.
        The function assumes that both vertices exists in the maze and the target is a neighbor of the source.
        As above, it can be verified using `assert source in get_vertices(graph)` and `assert target in get_neighbors(source, graph)` but at some cost.
        In:
            * source: Source vertex in the graph.
            * target: Target vertex, assumed to be a neighbor of the source vertex in the graph.
            * graph:  Graph on which to get the weight from the source vertex to the target vertex.
        Out:
            * weight: Weight of the corresponding edge in the graph.
    """
    
    # If "maze_representation" option is set to "dictionary"
    if isinstance(graph, dict):
        weight = graph[source][target]
    
    # If "maze_representation" option is set to "matrix"
    elif isinstance(graph, numpy.ndarray):
        weight = graph[source, target]
    
    # Unhandled data type
    else:
        raise Exception("Unhandled graph type", type(graph))
    
    # Done
    return weight

#####################################################################################################################################################
############################################################# GRAPH TRAVERSAL FUNCTIONS #############################################################
#####################################################################################################################################################

def traversal ( source:             int,
                graph:              Union[numpy.ndarray, Dict[int, Dict[int, int]]],
                create_structure:   Callable[[], Any],
                push_to_structure:  Callable[[Any, Tuple[int, int, int]], None],
                pop_from_structure: Callable[[Any], Tuple[int, int, int]],
                early_stop_vertex:  int = None
              ) ->                  Tuple[Dict[int, int], Dict[int, Union[None, int]]]:

    """
        Traversal function that explores a graph from a given vertex.
        This function is generic and can be used for most graph traversal.
        To adapt it to a specific traversal, you need to provide the adapted functions to create, push and pop elements from the structure.
        In:
            * source:             Vertex from which to start the traversal.
            * graph:              Graph on which to perform the traversal.
            * create_structure:   Function that creates an empty structure to use in the traversal.
            * push_to_structure:  Function that adds an element of type B to the structure of type A.
            * pop_from_structure: Function that returns and removes an element of type B from the structure of type A.
            * early_stop_vertex:  If set to a vertex, the traversal stops when it is first encountered.
        Out:
            * distances_to_explored_vertices: Dictionary where keys are explored vertices and associated values are the lengths of the paths to reach them.
            * routing_table:                  Routing table to allow reconstructing the paths obtained by the traversal.
    """
    
    # Initialize a data structure with start vertex
    queuing_structure = create_structure()
    push_to_structure(queuing_structure, (0, source, None))
    
    # Explore the graph
    distances_to_explored_vertices = {}
    routing_table = {} 
    while len(queuing_structure) > 0:
    
        # Get an unexplored element from the structure
        (distance_to_current_vertex, current_vertex, parent) = pop_from_structure(queuing_structure)
        if current_vertex not in distances_to_explored_vertices:
        
            # It is now explored
            distances_to_explored_vertices[current_vertex] = distance_to_current_vertex
            routing_table[current_vertex] = parent
            if current_vertex == early_stop_vertex:
                break
            
            # Add its neighbors to the structure for later exploration
            for neighbor in get_neighbors(current_vertex, graph):
                distance_to_neighbor = distance_to_current_vertex + get_weight(current_vertex, neighbor, graph)
                push_to_structure(queuing_structure, (distance_to_neighbor, neighbor, current_vertex))
    
    # Once all vertices have been explored, it is over
    return distances_to_explored_vertices, routing_table

#####################################################################################################################################################

def bfs ( source:             int,
          graph:              Union[numpy.ndarray, Dict[int, Dict[int, int]]],
          early_stop_vertex:  int = None
        ) ->                  Tuple[Dict[int, int], Dict[int, Union[None, int]]]:

    """
        A BFS is a particular traversal where vertices are explored in the order where they are added to the structure.
        In:
            * source:             Vertex from which to start the traversal.
            * graph:              Graph on which to perform the traversal.
            * early_stop_vertex:  If set to a vertex, the traversal stops when it is first encountered.
        Out:
            * distances_to_explored_vertices: Dictionary where keys are explored vertices and associated values are the lengths of the paths to reach them.
            * routing_table:                  Routing table to allow reconstructing the paths obtained by the traversal.
    """
    
    # Function to create an empty FIFO, encoded as a list
    def _create_structure ():
        return []

    # Function to add an element to the FIFO (elements enter by the end)
    def _push_to_structure (structure, element):
        structure.append(element)

    # Function to extract an element from the FIFO (elements exit by the beginning)
    def _pop_from_structure (structure):
        return structure.pop(0)
    
    # Perform the traversal
    distances_to_explored_vertices, routing_table = traversal(source, graph, _create_structure, _push_to_structure, _pop_from_structure, early_stop_vertex)
    return distances_to_explored_vertices, routing_table

#####################################################################################################################################################

def find_route ( routing_table: Dict[int, Union[None, int]],
                 source:        int,
                 target:        int
               ) ->             List[int]:

    """
        Function to return a sequence of locations using a provided routing table.
        In:
            * routing_table: Routing table as obtained by the traversal.
            * source:        Vertex from which we start the route (should be the one matching the routing table).
            * target:        Target to reach using the routing table.
        Out:
            * route: Sequence of locations to reach the target from the source, as perfomed in the traversal.
    """
    
    # We check precessors along the search
    route = [target]
    while route[0] != source:
        route.insert(0, routing_table[route[0]])
    return route

#####################################################################################################################################################
############################################################### GAME RELATED FUNCTIONS ##############################################################
#####################################################################################################################################################

def locations_to_action ( source:     int,
                          target:     int,
                          maze_width: int
                        ) ->          str: 

    """
        Function to transform two locations into an action to reach target from the source.
        In:
            * source:     Vertex on which the player is.
            * target:     Vertex where the character wants to go.
            * maze_width: Width of the maze in number of cells.
        Out:
            * action: Name of the action to go from the source to the target.
    """

    # Convert indices in row, col pairs
    source_row = source // maze_width
    source_col = source % maze_width
    target_row = target // maze_width
    target_col = target % maze_width
    
    # Check difference to get direction
    difference = (target_row - source_row, target_col - source_col)
    if difference == (0, 0):
        action = "nothing"
    elif difference == (0, -1):
        action = "west"
    elif difference == (0, 1):
        action = "east"
    elif difference == (1, 0):
        action = "south"
    elif difference == (-1, 0):
        action = "north"
    else:
        raise Exception("Impossible move from", source, "to", target)
    return action

#####################################################################################################################################################

def locations_to_actions ( locations:  List[int],
                           maze_width: int
                         ) ->          List[str]: 

    """
        Function to transform a list of locations into a list of actions to reach vertex i+1 from vertex i.
        In:
            * locations:  List of locations to visit in order.
            * maze_width: Width of the maze in number of cells.
        Out:
            * actions: Sequence of actions to visit the list of locations.
    """
    
    # We iteratively transforms pairs of locations in the corresponding action
    actions = []
    for i in range(len(locations) - 1):
        action = locations_to_action(locations[i], locations[i + 1], maze_width)
        actions.append(action)
    return actions

#####################################################################################################################################################

def get_opponent_name ( teams: Dict[str, List[str]],
                        name:  str
                      ) ->     str:

    """
        Function to return the opponent's name.
        It assumes that there are only two players in the game, each in a distinct team.
        In:
            * teams: Teams of players.
            * name:  Name of the current player.
        Out:
            * opponent_name: Name of the opponent.
    """
    
    # Find the other player
    opponent_team = [team for team in teams if name not in teams[team]][0]
    opponent_name = teams[opponent_team][0]
    return opponent_name

#####################################################################################################################################################
#####################################################################################################################################################
