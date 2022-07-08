import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def make_df(filename):
    df = pd.read_csv(filename)
    def lm(value):
        try:
            return int(value)
        except:
            return 0
    df['life_main'] = df['life_main'].apply(lm)

    result = None
    try:
        result = df['result']
    except:
        pass
    new_df = {
        "id": df['id'],
        "life_main": df['life_main']
    }
    if result is not None:
        new_df["result"] = result
    return pd.DataFrame(new_df)

df_train = make_df("train 2.csv")
x = df_train.drop('result', axis = 1)  # только данные о пользоватеях сайта
y = df_train['result']  # только целевой результат

sc = StandardScaler()
x = sc.fit_transform(x)  # смасштабировали данные о пользователях
classifier = KNeighborsClassifier(n_neighbors=3)  # создали сеть
classifier.fit(x, y)  # Обучили сеть

df_test = make_df("test 2.csv")
x_pred = sc.fit_transform(df_test)  # смасштабировали данные о пользователях
y_pred = classifier.predict(x_pred)  # Спрогнозировали купят/не купят

df_pred = pd.DataFrame({"id": df_test["id"], "result": y_pred})
df_pred.to_csv("res.csv", index=False)

