import os
import neat
import pandas as pd
import utils
import numpy as np
import visualize
import pickle
import time
import random

START_TIME = time.perf_counter()

# File directory
local_dir = os.path.dirname(__file__)
out_dir = os.path.join(local_dir, 'out')
out_dir = os.path.join(out_dir, 'single_pole')

# Read games and predictions
# data = pd.read_csv('Current Stats and Games.csv')
data = pd.read_csv('NE_input_df.csv') # Use Keras predictions as inputs

# data = data.iloc[:,-5000:] #Use only the last 5000 games
questions = data[['Home Odds', 'Vis Odds', 'Current Preds', 'Prev Preds', 'Ensemble', 'Home Win']]


MAX_BET_SIZE = 10
MIN_ODDS_SIZE = 0
MAX_ODDS_SIZE = 2.0
N_GENERATIONS = 70
QUIZ_LENGTH = 50
QUIZ_START = random.randint(1, len(data) - QUIZ_LENGTH)
# quiz_length = len(data) # Train on all data?


# Get population size
with open('config.txt', 'r') as f:
    data = f.read()
    pop = data.find(('pop_size'))
    pop_size = int(data[pop+11:pop+14])


bet_sizes = []
no_bets_list = []
outcomes = []

# Bet dictionary
bet_dict = {
    0: 'True',
    1: 'False',
    2: 'No Bet'
}

# NEAT QUIZ
def run_quiz(net, question):

    answer = question[-1]
    outcome = 0
    guess = net.activate(question[:5])
    # [0] = Home bet, [1] = Away bet, [2] =  No bet, [3] = bet size

    # Find max output index of: 0, 1, 2
    bets = guess[:2]
    bet_index = bets.index(max(bets))
    bet = bet_dict[bet_index]

    # Bet Size is 3rd index
    bet_size = round((guess[3] * MAX_BET_SIZE), 2)

    # Check for upper and lower limit of bet sizes
    bet_size = MAX_BET_SIZE if bet_size > MAX_BET_SIZE else 0 if bet_size < 0 else bet_size
    bet_sizes.append(bet_size)
    

    if (bet == 'True') and (str(answer) == 'True') and (question[0] > MIN_ODDS_SIZE) and (question[0] < MAX_ODDS_SIZE):
        outcome += (question[0] * bet_size) - bet_size
    elif (bet == 'False') and (str(answer) == 'False') and (question[1] > MIN_ODDS_SIZE) and (question[1] < MAX_ODDS_SIZE):
        outcome += (question[1] * bet_size) - bet_size
    elif (bet == 'No Bet') or (bet == 'True' and question[0] <= MIN_ODDS_SIZE) or (bet == 'False' and question[1] <= MIN_ODDS_SIZE) \
        or (bet == 'True' and question[0] >= MAX_ODDS_SIZE) or (bet == 'False' and question[1] >= MAX_ODDS_SIZE):
        outcome = 0
        no_bets_list.append(bet)
    else:
        outcome -= bet_size
    outcomes.append(outcome)
    return outcome



def eval_fitness(net, questions):
    fitness = 0
    questions_sample = questions.iloc[QUIZ_START:(QUIZ_START + QUIZ_LENGTH)] # Games in historic order
    questions_sample = questions_sample.values.tolist()
    for i in range(QUIZ_LENGTH):
        fitness += run_quiz(net, questions_sample[i])
    return fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = eval_fitness(net, questions)


def run_experiment(config_file, N_GENERATIONS):
    '''
    Function to run experiement
    Winner genome will render a graph and stats
    Args:
        config_file: path to the file with experiment config
    '''
    # Load config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    
    # Create population
    p = neat.Population(config)

    # Stats reporting
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix='out/spb-neat-checkpoint-'))

    # Run for N generations
    best_genome = p.run(eval_genomes, n=N_GENERATIONS)
    
    # Save winner
    with open("winner.pkl", "wb") as f:
        pickle.dump(best_genome, f)
        f.close()

    # Visualise Experiment results
    node_names = {-1:'Home Odds', -2:'Away Odds', -3:'Ensemble', 0:'Home Bet', 1: 'Away Bet', 2:'No Bet', 3: 'Bet Size'}
    visualize.draw_net(config, best_genome, True, node_names=node_names, directory=out_dir)
    visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join(out_dir, 'avg_fitness.svg'))
    visualize.plot_species(stats, view=True, filename=os.path.join(out_dir, 'speciation.svg'))
    


if __name__=='__main__':
    # Path to config file
    config_path = os.path.join(local_dir, 'config.txt')

    utils.clear_output(out_dir)

    # run experiment
    run_experiment(config_path, N_GENERATIONS)

    # Print report
    print('Max bet size: ', max(bet_sizes))
    print('Min bet size: ', min(bet_sizes))
    print('Avg bet size: ', np.mean(bet_sizes))
    print('Avg Games bet on: ', ((len(bet_sizes) / N_GENERATIONS) / pop_size))
    print('Avg Games not bet on: ', ((len(no_bets_list) / N_GENERATIONS) / pop_size))
    print('Percent of games not bet on: ', round(((((len(no_bets_list) / N_GENERATIONS) / pop_size) / QUIZ_LENGTH)) * 100, 2), '%')

    # Timing
    end_time = time.perf_counter()
    elapsed_time = round(end_time - START_TIME, 2)
    print('Elapsed time: ', elapsed_time, ' seconds')