import NeuralNetwork as Neural
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime as rtc
from sklearn.preprocessing import StandardScaler


def train_all(x : np.ndarray, y : np.ndarray, dataframe_name: str, epochs: int ) -> None:
    functions = ["relu","leaky_relu", "tanh", "sigmoid"]
    for function in functions:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        y_train = y_train.to_numpy().reshape(-1,1)
        y_test = y_test.to_numpy().reshape(-1,1)

        layer_structure = [x_train.shape[1],64,64,64,64,1]

        nn = Neural.NeuralNetwork(layer_structure, epochs, 64, 1e-5, 0.2, 1, function)
        nn._fit(x_train, y_train)

        y_pred = nn.predict(x_test)
        nn.plot_learning()

        error = np.abs(np.mean((y_test-y_pred)/y_pred*100))
        logs = open(f"training_logs/{function}_logs.txt", 'a')
        logs.write(f"{dataframe_name}:Test error: {error}%: End: {rtc.now()}\n")
        logs.close()

    plt.title(f"Dataframe name: {dataframe_name}. Epochs: {epochs}")
    plt.legend()
    plt.savefig(f"graphs/{dataframe_name}/{dataframe_name}_{epochs}_epochs.png") 
    plt.clf()


def set_dataframe_and_train_the_model(dataframe_name : str, traget_name : str, epochs: int) -> None:
    df = pd.read_csv(f'data/{dataframe_name}.csv')
    x = df.drop(f'{traget_name}', axis = 1)
    y = df[f'{traget_name}']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    train_all(x,y,dataframe_name, epochs)


dataframes = [["age_predictions_cleaned","age_group"],["breast_cancer","diagnosis"],["diabetes_prediction_dataset","diabetes"],["Raisin_Dataset","Class"], ["student_performance","FinalGrade"]]
for dataframe in dataframes:
    for i in range(20, 1001, 20):
        set_dataframe_and_train_the_model(dataframe[0], dataframe[1], i)
    print(f"The training has finished for {dataframe[0]}")

print("The training has finished successfully.")