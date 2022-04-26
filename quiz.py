import os
import neat
import pandas as pd
import utils
import numpy as np
import visualize
import pickle


local_dir = os.path.dirname(__file__)
out_dir = os.path.join(local_dir, 'out')
out_dir = os.path.join(out_dir, 'single_pole')

data = pd.read_csv('Current Stats and Games.csv')
answers = data['Home Win']
questions = data[['Home Odds', 'Vis Odds']]
questions = questions.values.tolist()
quiz_length = len(data)

bet_sizes = []
no_bets_list = []
outcomes = []

### NEAT QUIZ
def run_quiz(net, question, answer):
    outcome = 0
    guess = net.activate(question)
    # guess is an array of 4
    # [0] = Home bet, [1] = Away bet, [2] =  No bet, [3] = bet size

    # Find H, A or NB
    bets = guess[:2]
    bet = bets.index(max(bets))
    bet_size = round((guess[3] * 10), 2)
    if bet_size > 0:
        bet_sizes.append(bet_size)

    if bet == 0 and str(answer) == 'True':
        outcome += question[0] * bet_size - bet_size
    elif bet == 1 and str(answer) == 'False':
        outcome += question[1] * bet_size - bet_size
    elif bet == 3:
        outcome = 0
        no_bets_list.append(bet)
    else:
        outcome -= bet_size
    outcomes.append(outcome)
    return outcome



def eval_fitness(net):
    fitness = 0
    for i in range(quiz_length):
        fitness += run_quiz(net, questions[i], answers[i])
    return fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = eval_fitness(net)


def run_experiment(config_file, n_generations=100):
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
    best_genome = p.run(eval_genomes, n=n_generations)
    
    # Save wommer
    with open("winner.pkl", "wb") as f:
        pickle.dump(best_genome, f)
        f.close()

    # Visualise Experiment results
    node_names = {-1:'A', -2:'B', 0:'A XOR B'}
    visualize.draw_net(config, best_genome, True, node_names=node_names, directory=out_dir)
    visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join(out_dir, 'avg_fitness.svg'))
    visualize.plot_species(stats, view=True, filename=os.path.join(out_dir, 'speciation.svg'))


if __name__=='__main__':
    # Path to config file
    config_path = os.path.join(local_dir, 'config.ini')

    utils.clear_output(out_dir)

    # run experiment
    run_experiment(config_path)
    print('Max bet size: ', max(bet_sizes))
    print('Min bet size: ', min(bet_sizes))
    print('Avg bet size: ', np.mean(bet_sizes))
    print('Games not bet on: ', len(no_bets_list))
    print('Percent of games not bet on: ', round(((len(no_bets_list) / len(data)) * 100), 2), '%')

# PLAYER INPUT

# def check_guess(question, answer):
#     outcome = 0
#     print(question)
#     guess = input('True or False?\n')
#     if str(guess) == 'True' and str(answer) == 'True':
#         print('CORRECT! | Home Team Won')
#         outcome += question[0] - 1
#     elif str(guess) == 'False' and str(answer) == 'False':
#         print('CORRECT! | Visitor Team Won')
#         outcome += question[1] - 1
#     else:
#         print('Incorrect')
#         outcome -= 1
#     return outcome





## USE THE FOLLOWING CODE TO MAKE SAY HOW TO BET ON FUTURE GAMES
# def replay_genome(config_path, genome_path="winner.pkl"):
#     # Load requried NEAT config
#     config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

#     # Unpickle saved winner
#     with open(genome_path, "rb") as f:
#         genome = pickle.load(f)

#     # Convert loaded genome into required data structure
#     genomes = [(1, genome)]

#     # Call game with only the loaded genome
#     game(genomes, config)


def make_bet(net, question, future_bets):
    guess = net.activate(question)
    # guess is an array of 4
    # [0] = Home bet, [1] = Away bet, [2] =  No bet, [3] = bet size

    # Find H, A or NB
    bets = guess[:2]
    bet = bets.index(max(bets))
    bet_size = round((guess[3] * 10), 2)

    if bet_size > 0:
        bet_sizes.append(bet_size)
    if bet == 0:
        future_bets.append('Home')
    elif bet == 1:
        future_bets.append('Away')
    elif bet == 3:
        future_bets.append('No bet')
        no_bets_list.append(bet)
    return future_bets

def make_bets(net):
    future_bets = []
    for i in range(quiz_length):
        make_bet(net, questions[i], future_bets)
    return future_bets

# future_bet_genome = load genome
# make_bets(future_bet_genome)


# Add the bets from the NE to the future games dataframe
# future_games_df['Action'] = all_bets