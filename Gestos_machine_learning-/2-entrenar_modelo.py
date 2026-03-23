import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# cargar dataset
data = pd.read_csv("gestos.csv", header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# crear modelo
model = RandomForestClassifier(n_estimators=200)

# entrenar
model.fit(X_train, y_train)

# accuracy general
accuracy = model.score(X_test, y_test)
print("Accuracy general:", round(accuracy*100,2), "%")

# reporte por gesto
print("\nConfianza por gesto:")
print(classification_report(y_test, model.predict(X_test)))

# guardar modelo
joblib.dump(model, "modelo_gestos.pkl")