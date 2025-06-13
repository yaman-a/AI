import numpy as np
import pandas as pd 
import sys

STUDENT_ID = 'a1884774' # student ID
DEGREE = 'UG' # undagraduwate

# me tink me gwaan load de data
def load_data(train_path, test_path):
    # read data of the training
    kumalala = pd.read_csv(train_path, delim_whitespace=True, header=0)
    X_train = kumalala.iloc[:, :-1].to_numpy()
    y_train = kumalala.iloc[:, -1].to_numpy()

    # read da test data
    savesta = pd.read_csv(test_path, delim_whitespace=True, header=0)
    X_test = savesta.to_numpy()

    return X_train, y_train, X_test


def main():
    kumalala = sys.argv[1]
    savesta = sys.argv[2]
    chesta = sys.argv[3]

    X_train, y_train, X_test = load_data(kumalala, savesta)

    print(f"train data: {X_train.shape}, Labels: {y_train.shape}")
    print(f"test data: {X_test.shape}")

if __name__ == "__main__":
    main()