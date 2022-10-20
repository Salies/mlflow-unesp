# Databricks notebook source
# MAGIC %sql
# MAGIC SELECT * FROM sandbox_apoiadores.abt_dota_pre_match

# COMMAND ----------

sdf = spark.table("sandbox_apoiadores.abt_dota_pre_match")
df = sdf.toPandas()

# COMMAND ----------

target_column = 'radiant_win'
id_column = 'match_id'
features_columns = list(set(df.columns.tolist()) - set([target_column, id_column]))
y = df[target_column]
X = df[features_columns]

# COMMAND ----------

from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

print("Número de linhas em X_train", X_train.shape[0])
print("Número de linhas em X_test", X_test.shape[0])
print("Número de linhas em X_train", y_train.shape[0])
print("Número de linhas em X_test", y_test.shape[0])

# COMMAND ----------

from sklearn import ensemble
import mlflow
from sklearn.neural_network import MLPClassifier
#model = GaussianNB()
#model = CalibratedClassifierCV(base_model)
#model.fit(X_train, y_train)

# COMMAND ----------

from sklearn import metrics

y_train_pred = model.predict(X_train)
y_train_prob = model.predict_proba(X_train)

acc_train = metrics.accuracy_score(y_train, y_train_pred)
print("Acurácia em treino:", acc_train)

# COMMAND ----------

y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)

acc_test = metrics.accuracy_score(y_test, y_test_pred)
print("Acurácia em teste:", acc_test)

# COMMAND ----------

mlflow.set_experiment("/Users/nahksalies@gmail.com/dota-unesp-serezane")

# COMMAND ----------

with mlflow.start_run():
    mlflow.sklearn.autolog()
    # 12, 0: 83% em treino e 61% em teste
    #model = ensemble.RandomForestClassifier(n_estimators=100, max_depth=12, random_state=0) # 60.89 em teste // depth 7 60.74
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 20), random_state=1)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    acc_train = metrics.accuracy_score(y_train, y_train_pred)
    print("Acurácia em treino:", acc_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)
    acc_test = metrics.accuracy_score(y_test, y_test_pred)
    print("Acurácia em teste:", acc_test)
    

# COMMAND ----------


