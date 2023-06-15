import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Importing the dataset
dataset = pd.read_csv('Data_Skripsi_2023.csv', sep=";")

# Pisahkan kolom target (y) dan atribut (X)
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

# ... (code for preprocessing, model training, and prediction)

# Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set 
classifier = GaussianNB() 
model = classifier.fit(X_train,y_train)

# Prediksi Test set results 
y_pred = classifier.predict(X_test)

# Dictionary mapping team codes to team name
team_mapping = {
    1: 'Roma',
    2: 'Inter',
    3: 'Benevento',
    4: 'Spal',
    5: 'Genoa',
    6: 'Crotone',
    7: 'Napoli',
    8: 'Fiorentina',
    9: 'Chievo',
    10: 'Sampdoria',
    11: 'Bologna',
    12: 'Cagliari',
    13: 'Udinese',
    14: 'Sasauolo',
    15: 'Verona',
    16: 'Torino',
    17: 'Lazio',
    18: 'Atalanta',
    19: 'Parma',
    20: 'Lecce',
    21: 'Empoli',
    22: 'Juventus',
    23: 'Milan',
    24: 'Frosinone',
    25: 'Brescia',
    26: 'Spezia',
    27: 'Salernitana',
    28: 'Venezia',
}

home_team = dataset.iloc[X_test[:, 0], 0].map(team_mapping).values
away_team = dataset.iloc[X_test[:, 1], 1].map(team_mapping).values
result = np.column_stack((home_team, away_team, y_pred))

# Define Streamlit app
def main():
    st.title("Aplikasi Prediksi Pertandingan")

    # Show historical match results
    st.subheader("Historical Match Results")
    df_result = history()
    st.dataframe(df_result)

    # Show final standings
    st.subheader("Final Standings")
    standings = get_final_standings()
    st.write(standings)

# Function to get historical match results
def history():
    home_team = dataset.iloc[X_test[:, 0], 0].map(team_mapping).values
    away_team = dataset.iloc[X_test[:, 1], 0].map(team_mapping).values
    result = np.column_stack((home_team, away_team, y_pred))

    df_result = pd.DataFrame(result, columns=['Home Team', 'Away Team', 'Result'])
    return df_result

# Function to calculate final standings
def get_final_standings():
    points = {}
    result = history().values
    for i in range(len(result)):
        home = result[i][0]
        away = result[i][1]
        pred = result[i][2]
        if home not in points:
            points[home] = 0
        if away not in points:
            points[away] = 0
        if pred == 0:
            points[home] += 0
            points[away] += 3
        elif pred == 1:
            points[home] += 1
            points[away] += 1
        else:
            points[home] += 3
            points[away] += 0

    sorted_points = sorted(points.items(), key=lambda x: x[1], reverse=True)
    standings = []
    for i, team in enumerate(sorted_points):
        standings.append(f"{i+1}. {team[0]} - {team[1]} points ")

    return "\n".join(standings)

if __name__ == '__main__':
    main()
