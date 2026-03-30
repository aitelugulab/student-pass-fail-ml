import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Create dataset
data = {
    'marks': [35, 40, 50, 60, 70, 20, 90, 30],
    'result': [0, 0, 1, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Step 2: Split data
X = df[['marks']]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 3: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Accuracy check
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Step 5: Take user input
marks = int(input("Enter student marks: "))

input_data = pd.DataFrame([[marks]], columns=['marks'])
prediction = model.predict(input_data)

# Step 7: Output result
if prediction[0] == 1:
    print("Result: PASS")
else:
    print("Result: FAIL")
