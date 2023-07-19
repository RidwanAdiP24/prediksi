import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

# Mapping team 
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

# Define Streamlit app
def main():
    st.title("Prediksi Klasemen dan Pertandingan Pada Liga Serie A")
    st.write("Format yang digunakan yaitu : HT, AT, FTHG, FTAG, HTHG, HTAG, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR dan FTR")
    st.write("Unggah file CSV dengan format yang sesuai.")

    # File uploader
    uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file, sep=";")
        
        # Pisah kolom label (y) dan atribut (X)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Inisialisasi LabelEncoder
        le = LabelEncoder()

        # Mengubah kolom HomeTeam pada kedua dataset menjadi label encoding dengan LabelEncoder yang sama
        X_train[:, 0] = le.fit_transform(X_train[:, 0])
        X_test[:, 0] = le.transform(X_test[:, 0])

        # Mengubah kolom AwayTeam pada kedua dataset menjadi label encoding dengan LabelEncoder yang sama
        X_train[:, 1] = le.fit_transform(X_train[:, 1])
        X_test[:, 1] = le.transform(X_test[:, 1])

        # Fitting Naive Bayes to the Training set
        classifier = GaussianNB()
        model = classifier.fit(X_train, y_train)

        # Prediksi Test set results
        y_pred = classifier.predict(X_test)

        result_df = pd.DataFrame({"Home Team": dataset.iloc[X_test[:, 0], 0].map(team_mapping).values,
                                  "Away Team": dataset.iloc[X_test[:, 1], 0].map(team_mapping).values,
                                  "Result": y_pred})

        # Calculate final points by team
        points = {}
        for i in range(len(result_df)):
            home = result_df.iloc[i]["Home Team"]
            away = result_df.iloc[i]["Away Team"]
            pred = result_df.iloc[i]["Result"]
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
        st.subheader("Final Standings")
        for i, team in enumerate(sorted_points):
            st.write(f"{i+1}. {team[0]} - {team[1]} points")
        
        # Print predicted results
        st.subheader("Prediksi Hasil Pertandingan")
        st.text("1 = Home Win")
        st.text("2 = Away Win")
        st.text("0 = Draw")
        result_df = pd.DataFrame({"Home Team": dataset.iloc[X_test[:, 0], 0].map(team_mapping).values,
                                  "Away Team": dataset.iloc[X_test[:, 1], 0].map(team_mapping).values,
                                  "Result": y_pred})
        st.dataframe(result_df)

        # Evaluate model
        #st.subheader("Evaluasi Model")
        #cm = confusion_matrix(y_test, y_pred)
        #st.write("Confusion Matrix:")
        #st.write(cm)
        #st.write("Classification Report:")
        #st.write(classification_report(y_test, y_pred))
        accuracy = accuracy_score(y_test, y_pred) * 100
        #st.write(f"Accuracy: {accuracy:.2f}%")

        # Cross Validation
        #st.subheader("Cross Validation")
        accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
        mean_accuracy = accuracies.mean() * 100
        #st.write(f"Mean Accuracy: {mean_accuracy:.2f}%")

if __name__ == '__main__':
    main()
