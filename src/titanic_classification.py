import os
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def main(): 
    start_time = time.time()

    for dirname, _, filenames in os.walk(os.path.join('..', 'input')):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    train_data = pd.read_csv(os.path.join("..", "input", "titanic", "train.csv"))
    train_data.head()

    test_data = pd.read_csv(os.path.join("..", "input", "titanic", "test.csv"))
    test_data.head()

    women = train_data.loc[train_data.Sex == 'female']["Survived"]
    rate_women = sum(women)/len(women)

    print("% of women who survived:", rate_women)

    men = train_data.loc[train_data.Sex == 'male']["Survived"]
    rate_men = sum(men)/len(men)

    print("% of men who survived:", rate_men)

    y = train_data["Survived"]

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('my_submission.csv', index=False)

    end_time = time.time()
    print(f"Notebook finished after about {round(end_time - start_time, 3)} s. ")

    print("Your submission was successfully saved!")

if __name__ == "__main__": 
    main()