#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This program aims at training the rat to beat a greedy opponent.
    To do so, we use reinforcement learning, and more precisely deep Q learning (DQN).
    If you set the TRAIN_MODEL constant to True, the rat will be training.
    Otherwise it will just try to get the cheese using what it has learned.

    In this lab, we will work on a full maze with no mud.
    The settings have been set to a 10x10 maze with 15 pieces of cheese.
    You can change these settings if you want.

    This file already provides you with a functional RL algorithm.
    However, it is not very efficient, and you will have to improve it.
    Before you do so, you should understand how it works.

    The file makes use of Weights and Biases (WandB) to monitor the training.
    You can disable its use by setting the USE_WANDB constant to False.
    If you want to use it, you will have to create an account on https://wandb.ai/ and get an API key.
    Then, you will have to store that key in a file named "wandb.key" and set the WANDB_KEY_PATH variable accordingly (by defaultit is set to the same directory as this file).

    Here is your mission:
        1 - Have a look at the code and make sure you understand it.
            Also, make sure the code runs for training (TRAIN_MODEL=True) and playing (TRAIN_MODEL=False).
            If not, you may have to install some packages or change some paths.
        2 - Improve the RL algorithm.
            There are multiple ways to do so, and you are free to choose the ones you want.
            Some ideas:
                * Improve the data representation.
                * Change the reward function.
                * Improve the exploration strategy (some keywords you may be looking for are "epsilon greedy" or "softmax policy" for instance).
                * Change the hyperparameters of the algorithm (discount factor, batch size, etc.).
                * Improve the model architecture.
                * Change the optimizer or its parameters.
                * Change the loss function.
                * Etc.
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Import PyRat
from pyrat import *

# External imports 
import torch.nn as nn
import torch
import random
import numpy
import shutil
import tqdm
import wandb
import sys
import os
import math
import torch.nn.functional as F

# Previously developed functions
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lab4"))
import greedy as opponent
from utils import get_opponent_name

import torch.optim.lr_scheduler as lr_scheduler

#####################################################################################################################################################
############################################################### CONSTANTS & VARIABLES ###############################################################
#####################################################################################################################################################

"""
    Paths where to store the stuff that must be shared from a game to the other.
"""
NAME = "RL_conv5"
OUTPUT_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), NAME)
MODEL_FILE_NAME = os.path.join(OUTPUT_DIRECTORY, "RL_model.pt")
OPTIMIZER_FILE_NAME = os.path.join(OUTPUT_DIRECTORY, "RL_optimizer.pt")
EXPERIENCE_FILE_NAME = os.path.join(OUTPUT_DIRECTORY, "RL_experience.pt")

#####################################################################################################################################################

"""
    Indicates if we are in train or test mode.
"""

SAVE_PARAMETERS = True
TRAIN_MODEL = True
RESET_TRAINING = True

#####################################################################################################################################################

"""
    RL algorithm parameters.
"""
QPLUS = False #new function with new parameter epsilon, allow to minimize the future reward
MAX_EXPERIENCE_SIZE = 100
EXPERIENCE_MIN_SIZE_FOR_TRAINING = 10
NB_BATCHES = 16
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1
CLIP_VALUE = 10

#####################################################################################################################################################

"""
    Parameters of the optimizer used to train the model.
"""

LOSS_FUNCTION = torch.nn.MSELoss()
LEARNING_RATE = 0.1

#####################################################################################################################################################

"""
    Number of PyRat games from which to learn.
"""

NB_EPISODES = 100

#####################################################################################################################################################

"""
    Wandb parameters.
"""

USE_WANDB = False
WANDB_KEY_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "wandb.key")

#####################################################################################################################################################
############################################################### DEEP Q LEARNING MODEL ###############################################################
#####################################################################################################################################################
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class DQN (torch.nn.Module):

    #############################################################################################################################################
    #                                                                CONSTRUCTOR                                                                #
    #############################################################################################################################################

    def __init__ ( self:              Self,
                   data_shape:        Tuple[int, ...],
                   actions_dimension: int
                 ) ->                 Self:

        """
            This function is the constructor of the class.
            In:
                * self:              Reference to the current object.
                * data_shape:        Shape of the input data.
                * actions_dimension: Number of possible actions.
            Out:
                * self: Reference to the current object.
        """
        # Inherit from parent class
        super(DQN, self).__init__()

        # Define the layers

        self.channels = data_shape[0]
        self.height = data_shape[1]
        self.widht = data_shape[2]

        # def conv_max(inp,out,stride):
        #     _layers = [
        #         torch.nn.Conv2d(in_channels=inp, out_channels=out, kernel_size=(3,3), stride=stride, padding=0, bias=False), 
        #         torch.nn.BatchNorm2d(out),
        #         torch.nn.ReLU(inplace=True), 
        #         torch.nn.MaxPool2d(stride)
        #         ]
        #     _layers = nn.Sequential(*_layers)
        #     return _layers
        
        def conv_mean(inp,out,stride):
            _layers = [
                torch.nn.Conv2d(in_channels=inp, out_channels=out, kernel_size=(3,3), stride=stride, padding=1, bias=False), 
                torch.nn.AvgPool2d(stride),
                torch.nn.BatchNorm2d(out),
                torch.nn.ReLU(inplace=True)
                ]
            _layers = nn.Sequential(*_layers)
            return _layers
        

        self.linear1 = torch.nn.Linear(256, 256)
        self.linear2 = torch.nn.Linear(256, actions_dimension)                          
                                        
        self.bn1 = torch.nn.BatchNorm2d(self.channels)

        self.conv1 = conv_mean(self.channels,32,1)
        self.conv2 = conv_mean(32,64,1)
        self.conv3 = conv_mean(64,128,1)
        self.conv4 = conv_mean(128,256,1)
        # self.conv5 = conv_mean(512,1024,1)

        """Initialisation of the weight"""
        self.init_weight()

        #sequential 
        
        # self.linear = torch.nn.Sequential(torch.nn.Flatten(),
        #                                   self.linear1,
        #                                   self.linear2)
        

        
    def init_weight(self):

        init_layer(self.linear1)
        init_layer(self.linear2)

        # init_bn(self.bn1)                     
        
    #############################################################################################################################################
    #                                                               PUBLIC METHODS                                                              #
    #########################################################################################################################################

    def forward ( self: Self,
                  x:    torch.tensor
                ) ->    torch.tensor:

        """
            This function performs a forward pass of the data through the model.
            In:
                * self: Reference to the current object.
                * x:    Input data.
            Out:
                * y: Output data.
        """
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.relu_(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        print("x.shape = ",x.shape)
        print("x= ",x)
        return x
#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def build_state ( maze:             Union[numpy.ndarray, Dict[int, Dict[int, int]]],
                  maze_width:       int,
                  maze_height:      int,
                  name:             str,
                  teams:            Dict[str, List[str]],
                  player_locations: Dict[str, int],
                  cheese:           List[int]
                ) ->                torch.tensor:

    """
        This function builds a state tensor to use as an input for the DQN.
        Here we assume a 2-player game.
        In:
            * maze:             Map of the maze, as data type described by PyRat's "maze_representation" option.
            * maze_width:       Width of the maze in number of cells.
            * maze_height:      Height of the maze in number of cells.
            * name:             Name of the player being trained.
            * teams:            Recap of the teams of players.
            * player_locations: Locations for all players in the game.
            * cheese:           List of available pieces of cheese in the maze.
        Out:
            * state: Tensor representing the state of the game.
    """
    
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Function to return an array with the player at the center
    def _center_maze (location):
        channel = torch.zeros(maze_height * 2 - 1, maze_width * 2 - 1, device=device)
        location_row, location_col = location // maze_width, location % maze_width
        for c in cheese:
            c_row, c_col = c // maze_width, c % maze_width
            channel[maze_height - 1 - location_row + c_row, maze_width - 1 - location_col + c_col] = 1
        return channel.unsqueeze(0)

    # A channel centered on the player
    player_channel = _center_maze(player_locations[name])

    # A channel centered on the opponent
    opponent_name = get_opponent_name(teams, name)
    opponent_channel = _center_maze(player_locations[opponent_name])

    # Concatenate channels
    state = torch.cat([player_channel, opponent_channel], dim=0).double()
    return state

#####################################################################################################################################################

def select_action ( state:              torch.tensor,
                    possible_actions:   List[str],
                    model:              DQN
                  ) ->                  str:

    """
        This function returns one of the possible actions, given the state of the game.
        In:
            * state:              State of the game in which we are now.
            * possible_actions:   Actions that the agent can perform.
            * model:              Deep Q learning model used to predict the next action.
        Out:
            * action: One action chosen from possible_actions.
    """

    # We use the model's prediction as a basis for the action
    with torch.no_grad():
        model_outputs = model(state.unsqueeze(0))

    # We get the best action
    action_index = torch.argmax(model_outputs).item()
    action = possible_actions[action_index]
    return action

#####################################################################################################################################################

def compute_reward ( state:                torch.tensor,
                     scores_at_state:      Dict[str, float],
                     new_state:            torch.tensor,
                     scores_at_new_state:  Dict[str, float],
                     name:                 str,
                     teams:                Dict[str, List[str]],
                     initial_cheese_count: int
                   ) ->                    float:
    

    """
        This function returns a reward for going from a state to another.
        Here we just return a negative constant, as we want to minimize the number of turns before gathering all cheese.
        In:
            * state:                State of the game before moving.
            * scores_at_state:      Scores of the players before moving.
            * new_state:            State of the game after moving.
            * scores_at_new_state:  Scores of the players after moving.
            * name:                 Name of the player being trained.
            * teams:                Recap of the teams of players.
            * initial_cheese_count: Number of cheese in the maze at the beginning of the game.
        Out:
            * reward: Value of the reward to return.
    """

    reward = 0.0
    
    # If there is no change in scores, we give no reward
    opponent_name = get_opponent_name(teams, name)
    if scores_at_state[name] == scores_at_new_state[name] and scores_at_state[opponent_name] == scores_at_new_state[opponent_name]:
        reward += 0.0

    if density(new_state,name) > density(state,name) and scores_at_state[name] == scores_at_new_state[name] and scores_at_state[opponent_name] == scores_at_new_state[opponent_name]:
        reward += 1
    # If we are tied with the opponent, and already half of the cheese is collected, we give no reward
    if scores_at_new_state[name] == scores_at_state[opponent_name] and scores_at_new_state[name] + scores_at_new_state[opponent_name] >= initial_cheese_count / 2:
        reward += 0.0
    
    # If we are tied with the opponent in the beginning of the game, we give a positive reward
    if scores_at_new_state[name] == scores_at_state[opponent_name] and scores_at_new_state[name] + scores_at_new_state[opponent_name] < initial_cheese_count / 2:
        reward += 1.0

    # If the opponent wins the game, we give a negative reward
    if scores_at_new_state[opponent_name] > initial_cheese_count / 2:
        reward = -2.0

    # If we win the game, we give a positive reward
    if scores_at_new_state[name] > initial_cheese_count / 2:
        reward += 4

    # If we caught a cheese alone, we give a positive reward
    if scores_at_new_state[name] > scores_at_state[name]:
        reward += 2
    
    # Otherwise, the opponent caught a cheese alone, and we give no reward
    return reward

#####################################################################################################################################################


def density(
            state:                torch.tensor,
            name:                 str,
             ) -> float:
    dist = numpy.zeros(state[0].shape)
    player_line = (state[0].shape[0]+1)/2
    player_col = (state[0].shape[1]+1)/2
    density = 0
    for line in range(state[0].shape[0]):
        for col in range(state[0].shape[1]):
                dist[line][col] = math.sqrt((player_line - line)**2 + (player_col - col)**2) 
                if dist[line][col] != 0:
                    density +=  1/(dist[line][col])

    

    return density



#########

def make_batch ( model:            DQN,
                 experience:       List[Dict[str, Any]],
                 possible_actions: List[str]
               ) ->                Tuple[torch.tensor, torch.tensor]:

    """
        This function builds batches from the memory to train the model on.
        Each batch is a pair (data, target), where each element has batch size as first dimension.
        In:
            * model:            Model to train.
            * experience:       List of experience situations encountered across games.
            * possible_actions: Actions mapped with the output of the model.    
        Out:
            * data:    Batch of data from the memory.
            * targets: Targets associated with the sampled data.
    """

    # Get indices
    batch_size = min(BATCH_SIZE, len(experience))
    indices = random.sample(range(len(experience)), batch_size)
    alpha = 1 - EPSILON
    # Create the batch
    data = torch.zeros(batch_size, *experience[0]["state"].shape)
    targets = torch.zeros(batch_size, len(possible_actions))
    for i in range(batch_size):
        
        # Data is the sampled state
        data[i] = experience[indices[i]]["state"]
        
        # Target is the discounted reward
        with torch.no_grad():
            targets[i] = model(data[i].unsqueeze(0))
            if experience[indices[i]]["over"]:
                targets[i, possible_actions.index(experience[indices[i]]["action"])] = experience[indices[i]]["reward"]
            else:
                model_outputs = model(experience[indices[i]]["new_state"].unsqueeze(0))
                if QPLUS:
                    targets[i, possible_actions.index(experience[indices[i]]["action"])] += alpha * (experience[indices[i]]["reward"] + DISCOUNT_FACTOR * torch.max(model_outputs).item() - targets[i, possible_actions.index(experience[indices[i]]["action"])])
                else :
                    targets[i, possible_actions.index(experience[indices[i]]["action"])] =  experience[indices[i]]["reward"] + DISCOUNT_FACTOR * torch.max(model_outputs).item() 

    # Done
    return data, targets
    
#####################################################################################################################################################

def train_model ( model:            DQN,
                  optimizer:        torch.optim.Optimizer,
                  experience:       List[Dict[str, Any]],
                  possible_actions: List[str],
                  
                ) ->                None:

    """
        This function trains the model on the experience.
        In:
            * model:            Model to train.
            * optimizer:        Optimizer used to train the model.
            * experience:       List of experience situations encountered across games.
            * possible_actions: Actions mapped with the output of the model.
        Out:
            * None.
    """
    
    # Define the loss function
    # loss_function = torch.nn.MSELoss()


    # Ensure model is in train mode
    model.train()
   
    # Train loop
    total_loss = 0
    for b in range(NB_BATCHES):
        
        # Create a random batch
        data, targets = make_batch(model, experience, possible_actions)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(data)
        loss = LOSS_FUNCTION(outputs, targets)
        loss.backward()
        if not USE_WANDB:
            print("loss = ", loss)
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)
        optimizer.step()
        # scheduler.step()
        #lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        
        # Accumulate total loss for debug
        total_loss += loss.item()

    #print(lr_list)
    # Log loss
    if USE_WANDB:
        wandb.log({"total_loss": total_loss})
    
#####################################################################################################################################################
##################################################### EXECUTED ONCE AT THE BEGINNING OF THE GAME ####################################################
#####################################################################################################################################################

def preprocessing ( maze:             Union[numpy.ndarray, Dict[int, Dict[int, int]]],
                    maze_width:       int,
                    maze_height:      int,
                    name:             str,
                    teams:            Dict[str, List[str]],
                    player_locations: Dict[str, int],
                    cheese:           List[int],
                    possible_actions: List[str],
                    memory:           threading.local
                  ) ->                None:

    """
        This function is called once at the beginning of the game.
        It is typically given more time than the turn function, to perform complex computations.
        Store the results of these computations in the provided memory to reuse them later during turns.
        To do so, you can crete entries in the memory dictionary as memory.my_key = my_value.
        In:
            * maze:             Map of the maze, as data type described by PyRat's "maze_representation" option.
            * maze_width:       Width of the maze in number of cells.
            * maze_height:      Height of the maze in number of cells.
            * name:             Name of the player controlled by this function.
            * teams:            Recap of the teams of players.
            * player_locations: Locations for all players in the game.
            * cheese:           List of available pieces of cheese in the maze.
            * possible_actions: List of possible actions.
            * memory:           Local memory to share information between preprocessing, turn and postprocessing.
        Out:
            * None.
    """
    
    # Instanciate a DQN model with weights loaded from file (if any)
    state_dimension = build_state(maze, maze_width, maze_height, name, teams, player_locations, cheese).shape
    actions_dimension = len(possible_actions)
    memory.model = DQN(data_shape= state_dimension, actions_dimension=actions_dimension)
    if os.path.exists(MODEL_FILE_NAME):
        memory.model.load_state_dict(torch.load(MODEL_FILE_NAME))
    
    # Instanciate an optimizer, initialized in its previous state to preserve momentum (if any)
    memory.optimizer = torch.optim.SGD(memory.model.parameters(), lr=LEARNING_RATE)

    if os.path.exists(OPTIMIZER_FILE_NAME):
        memory.optimizer.load_state_dict(torch.load(OPTIMIZER_FILE_NAME))

    # In train mode, load the experience from previous games (if any)
    memory.experience = []
    if TRAIN_MODEL:
        if os.path.exists(EXPERIENCE_FILE_NAME):
            memory.experience = torch.load(EXPERIENCE_FILE_NAME)

    # We remember the initial cheese count and previous turn scores
    memory.initial_cheese_count = len(cheese)
    memory.previous_scores = {p: 0 for p in player_locations}

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
    
    # Select an action to perform
    state = build_state(maze, maze_width, maze_height, name, teams, player_locations, cheese)
    action = select_action(state, possible_actions, memory.model)
    
    # Stuff to do if training the model
    if TRAIN_MODEL:
        
        # Remove old experience entries if needed
        if len(memory.experience) >= MAX_EXPERIENCE_SIZE:
            del memory.experience[0]
        
        # Complement the previous turn and initialize the current one
        if len(memory.experience) > 0 and "over" not in memory.experience[-1]:
            memory.experience[-1]["reward"] = compute_reward(memory.experience[-1]["state"], memory.previous_scores, state, player_scores, name, teams, memory.initial_cheese_count)
            memory.experience[-1]["new_state"] = state
            memory.experience[-1]["over"] = False
        memory.experience.append({"state": state, "action": action})
        

        # Save scores for next turn
        memory.previous_scores = player_scores

    # Done
    return action

#####################################################################################################################################################
######################################################## EXECUTED ONCE AT THE END OF THE GAME #######################################################
#####################################################################################################################################################

def postprocessing ( maze:             Union[numpy.ndarray, Dict[int, Dict[int, int]]],
                     maze_width:       int,
                     maze_height:      int,
                     name:             str,
                     teams:            Dict[str, List[str]],
                     player_locations: Dict[str, int],
                     player_scores:    Dict[str, float],
                     player_muds:      Dict[str, Dict[str, Union[None, int]]],
                     cheese:           List[int],
                     possible_actions: List[str],
                     memory:           threading.local,
                     stats:            Dict[str, Any],
                   ) ->                None:

    """
        This function is called once at the end of the game.
        It is not timed, and can be used to make some cleanup, analyses of the completed game, model training, etc.
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
            * None.
    """

    # Stuff to do if training the model
    if TRAIN_MODEL:
        
        # Complement the experience and save it
        state = build_state(maze, maze_width, maze_height, name, teams, player_locations, cheese)
        if len(memory.experience) > 0 and "over" not in memory.experience[-1]:
            memory.experience[-1]["reward"] = compute_reward(memory.experience[-1]["state"], memory.previous_scores, state, player_scores, name, teams, memory.initial_cheese_count)
            memory.experience[-1]["new_state"] = state
            memory.experience[-1]["over"] = True
            #print(state[0])
            #print(density(state,name))
        torch.save(memory.experience, EXPERIENCE_FILE_NAME)
        
        # Train the model if we have enough memory
        if len(memory.experience) >= EXPERIENCE_MIN_SIZE_FOR_TRAINING:
                # learning rate scheduler
            #scheduler = lr_scheduler.StepLR(memory.optimizer, step_size=16, gamma =0.999)
            train_model(memory.model, memory.optimizer, memory.experience, possible_actions)
            torch.save(memory.model.state_dict(), MODEL_FILE_NAME)
            torch.save(memory.optimizer.state_dict(), OPTIMIZER_FILE_NAME)

        # Log the final scores
        if USE_WANDB:
            opponent_name = get_opponent_name(teams, name)
            wandb.log({"final_score[player]": player_scores[name],
                       "final_score[opponent]": player_scores[opponent_name],
                       "final_score_difference": player_scores[name] - player_scores[opponent_name]})
    
#####################################################################################################################################################
######################################################################## GO! ########################################################################
#####################################################################################################################################################

if __name__ == "__main__":

    # Map the functions to the character
    players = [{"name": "RL",
                    "team": "You",
                    "skin": "rat",
                    "preprocessing_function": preprocessing,
                    "turn_function": turn,
                    "postprocessing_function": postprocessing},
               {"name": "Greedy",
                    "team": "Opponent",
                    "skin": "python",
                    "preprocessing_function": opponent.preprocessing if "preprocessing" in dir(opponent) else None,
                    "turn_function": opponent.turn,
                    "postprocessing_function": opponent.postprocessing if "postprocessing" in dir(opponent) else None}]

    # Customize the game elements
    config = {"maze_width": 21,
              "maze_height": 15,
              "cell_percentage": 100.0,
              "wall_percentage": 0.0,
              "mud_percentage": 0.0,
              "nb_cheese": 40}

    # Train mode
    if TRAIN_MODEL:
        
        # Remove old files if needed
        if RESET_TRAINING:
            if os.path.exists(OUTPUT_DIRECTORY):
                shutil.rmtree(OUTPUT_DIRECTORY, ignore_errors=True)
            os.mkdir(OUTPUT_DIRECTORY)
        
        if SAVE_PARAMETERS:
            text = ""
            text += "PARAMETERS :\n\r"
            text += "QPLUS : "+  str(QPLUS) + "\n\r"
            text += "LOSS_FUNCTION : "+  str(LOSS_FUNCTION) + "\n\r"
            text += "MAX_EXPERIENCE_SIZE : "+  str(MAX_EXPERIENCE_SIZE) + "\n\r"
            text += "EXPERIENCE_MIN_SIZE_FOR_TRAINING : "+  str(EXPERIENCE_MIN_SIZE_FOR_TRAINING) + "\n\r"
            text += "NB_BATCHES : "+  str(NB_BATCHES) + "\n\r"
            text += "BATCH_SIZE : "+  str(BATCH_SIZE) + "\n\r" 
            text += "DISCOUNT_FACTOR : "+  str(DISCOUNT_FACTOR) + "\n\r" 
            text += "EPSILON : "+  str(EPSILON) + "\n\r" 
            text += "LEARNING_RATE : "+  str(LEARNING_RATE) + "\n\r" 
            text += "NB_EPISODES : "+  str(NB_EPISODES) + "\n\r" 
            text += "\n\n"
            text += "Write here the shape of the neural network :\n\n"

            f = open(OUTPUT_DIRECTORY+'/parameters.txt', "w")
            f.write(text)
            f.close()

        
        # Connect to WandB for monitoring
        if USE_WANDB:
            wandb.login(key=open(WANDB_KEY_PATH).read().strip(), force=True)
            #wandb.login("81cb2f910cb9a6c5666fcc53cbfc75e51af4a01a", force=True)
            wandb.init(project="PyRat_RL", dir=OUTPUT_DIRECTORY)
        
        # Run multiple games with no GUI
        config["synchronous"] = True
        config["preprocessing_time"] = 0.0
        config["turn_time"] = 0.0
        config["render_mode"] = "no_rendering"
        for i in tqdm.tqdm(range(NB_EPISODES), desc="Episode", position=0, leave=False):
            game = PyRat(players, **config)
            stats = game.start()
            if stats == {}:
                break
            
    # Test mode
    else:
        
        # Make a single game with GUI
        game = PyRat(players, **config)
        stats = game.start()
        print(stats)

#####################################################################################################################################################
#####################################################################################################################################################
