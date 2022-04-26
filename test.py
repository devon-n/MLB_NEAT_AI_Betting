import pandas as pd
import numpy as np
import random

MAX_BET_SIZE = 10
MIN_ODDS_SIZE = 0
MAX_ODDS_SIZE = 2

no_bets_list = []
bet_sizes = []
bet_direction = []

# Read games and predictions
# data = pd.read_csv('Current Stats and Games.csv')
data = pd.read_csv('NE_input_df.csv') # Use Keras predictions as inputs

# data = data.iloc[:,-5000:] #Use only the last 5000 games
questions = data[['Home Odds', 'Vis Odds', 'Current Preds', 'Prev Preds', 'Ensemble', 'Home Win']]


MAX_BET_SIZE = 10
MIN_ODDS_SIZE = 0
MAX_ODDS_SIZE = 100
N_GENERATIONS = 70
QUIZ_LENGTH = 4
QUIZ_START = random.randint(1, len(data) - QUIZ_LENGTH)

questions = questions.iloc[QUIZ_START:(QUIZ_START + QUIZ_LENGTH)] # Games in historic order


bet_sizes = []
no_bets_list = []
outcomes = []
bet_directions = []

# Bet dictionary
bet_dict = {
    0: 'True',
    1: 'False',
    2: 'No Bet'
}

# NEAT QUIZ
def run_quiz(question):

    outcome = 0
    answer = question[-1]
    print(question)
    print(answer)
    bet = int(input('Home:0, Away: 1, No Bet: 2\n'))
    bet = bet_dict[bet]
    bet_size = 1
    
    

    if (bet == 'True') and (str(answer) == 'True') and (question[0] > MIN_ODDS_SIZE) and (question[0] < MAX_ODDS_SIZE):
        outcome += (question[0] * bet_size) - bet_size
    elif (bet == 'False') and (str(answer) == 'False') and (question[1] > MIN_ODDS_SIZE) and (question[1] < MAX_ODDS_SIZE):
        outcome += (question[1] * bet_size) - bet_size
    elif bet == 'No Bet' or (bet == 'True' and question[0] <= MIN_ODDS_SIZE) or (bet == 'False' and question[1] <= MIN_ODDS_SIZE) \
        or (bet == 'True' and question[0] >= MAX_ODDS_SIZE) or (bet == 'False' and question[1] >= MAX_ODDS_SIZE):
        outcome = 0
        no_bets_list.append(bet)
    else:
        outcome -= bet_size

    bet_direction.append(bet)
    bet_sizes.append(bet_size)
    outcomes.append(outcome)
    
    print(outcome)
    print()
    return outcome



def eval_fitness(questions):
    fitness = 0
    # questions_sample = questions.sample(quiz_length) # Random games
    # questions_sample = questions.iloc[QUIZ_START:(QUIZ_START + QUIZ_LENGTH)] # Games in historic order
    # questions_sample = questions_sample.values.tolist()
    questions_sample = questions.values.tolist()
    for i in range(len(questions_sample)):
        fitness += run_quiz(questions_sample[i])
    return fitness

eval_fitness(questions)

outcome_df = pd.DataFrame()

outcome_df[['Home Odds', 'Vis Odds', 'Home Win']] = questions[['Home Odds', 'Vis Odds', 'Home Win']]
outcome_df['Bet Direction'] = bet_direction
outcome_df['Bet Sizes'] = bet_sizes
outcome_df['Outcome'] = outcomes
outcome_df['Bankroll'] = outcome_df['Outcome'].cumsum()

outcome_df.to_csv('Test.csv', index=False)


### 