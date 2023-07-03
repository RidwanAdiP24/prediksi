import streamlit as st
# Import library
import pandas as pd
import numpy as np

# Import dataset
dataset = pd.read_csv('Data_Skripsi_2023.csv', sep=";")
print(dataset)


# Pisahkan kolom target (y) dan atribut (X)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#unique_home = np.unique(X[:, 0])
#print("Jumlah tim pada kolom HomeTeam:", len(unique_home))
#print("Tim pada kolom HomeTeam:", unique_home)

#unique_away = np.unique(X[:, 1])
#print("Jumlah tim pada kolom AwayTeam:", len(unique_away))
#print("Tim pada kolom AwayTeam:", unique_away)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.preprocessing import LabelEncoder

# Inisialisasi LabelEncoder
le = LabelEncoder()

# Mengubah kolom HomeTeam pada kedua dataset menjadi label encoding dengan LabelEncoder yang sama
X_train[:, 0] = le.fit_transform(X_train[:, 0])
X_test[:, 0] = le.transform(X_test[:, 0])

# Mengubah kolom AwayTeam pada kedua dataset menjadi label encoding dengan LabelEncoder yang sama
X_train[:, 1] = le.fit_transform(X_train[:, 1])
X_test[:, 1] = le.transform(X_test[:, 1])

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
model = classifier.fit(X_train,y_train)

# Prediksi Test set results
y_pred = classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, y_pred)*100)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print (accuracies.mean()*100)

# Dictionary mapping team codes to team name
team_mapping = {
    1: 'Udinese',
    2: 'Milan',
    3: 'Benevento',
    4: 'Napoli',
    5: 'Genoa',
    6: 'Crotone',
    7: 'Fiorentina',
    8: 'Lazio',
    9: 'Inter',
    10: 'Sampdoria',
    11: 'Bologna',
    12: 'Cagliari',
    13: 'Roma',
    14: 'Sasauolo',
    15: 'Verona',
    16: 'Torino',
    17: 'Spal',
    18: 'Juventus',
    19: 'Parma',
    20: 'Lecce',
    21: 'Empoli',
    22: 'Atalanta',
    23: 'Chievo',
    24: 'Frosinone',
    25: 'Brescia',
    26: 'Spezia',
    27: 'Salernitana',
    28: 'Venezia',
}
home_team = dataset.iloc[X_test[:, 0], 0].map(team_mapping).values
away_team = dataset.iloc[X_test[:, 1], 1].map(team_mapping).values
result = np.column_stack((home_team, away_team, y_pred))

df_result = pd.DataFrame(result, columns=['Home Team', 'Away Team', 'Result'])
print(df_result)

# Kalkulasi final point by team
points = {}
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

# Sorting teams by points
sorted_points = sorted(points.items(), key=lambda x: x[1], reverse=True)

# Print final standings
print("Final Standings:")
print("Team\tPoints")
for i in range(len(sorted_points)):
    print(f"{sorted_points[i][0]}\t{sorted_points[i][1]}")

# Define Streamlit app
def main():
    st.title(" Prediksi Pertandingan")
    # Show final standings
    st.subheader(" Prediksi Final Standing")
    standings = get_final_standings()
    st.write(standings)
    
    # Show historical match results
    st.subheader("Prediksi Histori Match Result")
    st.text("1 = Home Win")
    st.text("2 = Away Win")
    st.text("0 = Draw")
    df_result = history()
    st.dataframe(df_result)

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
