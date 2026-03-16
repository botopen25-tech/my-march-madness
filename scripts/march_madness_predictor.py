import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load historical March Madness data
try:
    # Data files extracted from Kaggle dataset
    historical_data = pd.read_csv('../data/NCAATourneyDetailedResults.csv')
    seeds = pd.read_csv('../data/NCAATourneySeeds.csv')
except FileNotFoundError:
    print("Data files not found. Please ensure the Kaggle dataset is downloaded and placed in ./data")
    raise

# Preprocess data

def preprocess_data(data, seeds):
    # Simplify seeding information
    seeds['SeedNumeric'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
    
    # Merge seeding info
    data = data.merge(seeds[['TeamID', 'Seed', 'SeedNumeric']], left_on='WTeamID', right_on='TeamID', suffixes=('', '_W'))
    data = data.merge(seeds[['TeamID', 'Seed', 'SeedNumeric']], left_on='LTeamID', right_on='TeamID', suffixes=('', '_L'))
    data['SeedDiff'] = data['SeedNumeric_W'] - data['SeedNumeric_L']
    
    # Focus on meaningful features
    features = data[['SeedDiff', 'WScore', 'LScore']]
    target = data['WTeamID']
    
    return features, target

# Main prediction function

def predict_tourney_outcomes():
    # Preprocess the data
    X, y = preprocess_data(historical_data, seeds)

    # Split the dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Add simulation logic here if needed

if __name__ == "__main__":
    print("Predicting tournament outcomes...")
    predict_tourney_outcomes()
