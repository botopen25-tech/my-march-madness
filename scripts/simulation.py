import pandas as pd
import numpy as np
from collections import defaultdict

# Load data
matchups = pd.read_csv('./my-march-madness/data/Tournament Matchups.csv')

# Preprocess data
def preprocess_data(data):
    data['Seed'] = data['SEED']
    data['Year'] = data['YEAR']
    return data

# Calculate win probability
def calculate_win_probability(seed1, seed2):
    if seed1 == seed2:
        return 0.5
    return seed2 / (seed1 + seed2)

# Simulate a single tournament
def simulate_tournament(data):
    results = defaultdict(lambda: {'Wins': 0, 'FinalFour': 0, 'TitleGame': 0, 'Champion': 0})
    game_results = []
    for year in data['Year'].unique():
        yearly_data = data[data['Year'] == year]
        for index, row in yearly_data.iterrows():
            team = row['TEAM']
            seed = row['Seed']
            opponents = yearly_data[yearly_data['TEAM'] != team]
            if not opponents.empty:
                opponent = opponents.sample(n=1).iloc[0]
                opponent_seed = opponent['Seed']
                win_prob = calculate_win_probability(seed, opponent_seed)
                if np.random.rand() < win_prob:
                    results[team]['Wins'] += 1
                    game_results.append({'Year': year, 'Game': f"{team} vs {opponent['TEAM']}", 'Winner': team, 'Probability': win_prob})
                    if row['ROUND'] <= 4:
                        results[team]['FinalFour'] += 1
                    if row['ROUND'] <= 2:
                        results[team]['TitleGame'] += 1
                    if row['ROUND'] == 1:
                        results[team]['Champion'] += 1
    return results, pd.DataFrame(game_results)

# Run simulations
def run_simulations(data, n=10000):
    summary_results = defaultdict(lambda: {'Wins': 0, 'FinalFour': 0, 'TitleGame': 0, 'Champion': 0})
    all_games = []
    for _ in range(n):
        results, game_details = simulate_tournament(data)
        all_games.append(game_details)
        for team, result in results.items():
            summary_results[team]['Wins'] += result['Wins']
            summary_results[team]['FinalFour'] += result['FinalFour']
            summary_results[team]['TitleGame'] += result['TitleGame']
            summary_results[team]['Champion'] += result['Champion']
    return summary_results, pd.concat(all_games)

# Summarize simulations
def summarize_simulations(summary_results, game_data, total_runs):
    df = pd.DataFrame.from_dict(summary_results, orient='index', columns=['Wins', 'FinalFour', 'TitleGame', 'Champion'])
    df = df / total_runs
    print(df.head())
    df.to_csv('./my-march-madness/simulation_results.csv')
    game_data.to_csv('./my-march-madness/game_simulations.csv', index=False)

if __name__ == "__main__":
    data = preprocess_data(matchups)
    summary_results, game_data = run_simulations(data)
    summarize_simulations(summary_results, game_data, 10000)