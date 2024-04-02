import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class FederatedClient:
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    def train(self):
        X, y = self.data
        self.model.fit(X, y)
    
    def get_model_update(self):
        return self.model.coef_, self.model.intercept_
    
    def set_model(self, model_params):
        self.model.coef_, self.model.intercept_ = model_params

class FederatedServer:
    def __init__(self, model):
        self.model = model
    
    def aggregate_updates(self, updates):
        avg_coef = np.mean([update[0] for update in updates], axis=0)
        avg_intercept = np.mean([update[1] for update in updates], axis=0)
        self.model.coef_ = avg_coef
        self.model.intercept_ = avg_intercept
    
    def get_model(self):
        return self.model.coef_, self.model.intercept_
    
    def set_model(self, model_params):
        self.model.coef_, self.model.intercept_ = model_params


def federated_learning(train_filenames, test_filenames):
    # Create a federated server with a linear regression model
    server = FederatedServer(LinearRegression())
    
    # Create a federated client for each node and store them in a list
    clients = []
    for filename in train_filenames:
        df_train = pd.read_csv(filename)
        X_train = df_train[['time_slot', 'electricity_usage']].values
        y_train = df_train['is_peak_usage'].values
        client = FederatedClient(LinearRegression(), (X_train, y_train))
        clients.append(client)
    
    # Perform federated learning
    print("Round 1:")
    
    # Initial training and update by the first client
    clients[0].train()
    aggregated_update = clients[0].get_model_update()
    print(f"    Client 1 has fully computed its weights: {aggregated_update[0]}, intercept: {aggregated_update[1]}")
    
    # Each subsequent client trains, receives the aggregated update, and adds its own update
    for i in range(1, len(clients)):
        clients[i].train()
        update = clients[i].get_model_update()
        print(f"    Client {i + 1} has fully computed its weights: {update[0]}, intercept: {update[1]}")
        aggregated_update = (
            aggregated_update[0] + update[0],
            aggregated_update[1] + update[1]
        )
        print(f"    Client {i} has sent its update to Client {i + 1}")
    
    # The last client averages the final aggregated update before sending it back to the server
    final_aggregated_update = (
        aggregated_update[0] / len(clients),
        aggregated_update[1] / len(clients)
    )
    server.set_model(final_aggregated_update)
    print(f"    Client {len(clients)} has sent the final aggregated update back to the server")
    print(f"    Final aggregated weights by Final Node: {server.model.coef_}, intercept: {server.model.intercept_}\n")
    
    # Return the final aggregated model from the server
    print(server.get_model())


     # Test the aggregated model on the test data
    test_data = []
    for filename in test_filenames:
        df_test = pd.read_csv(filename)
        X_test = df_test[['time_slot', 'electricity_usage']].values
        y_test = df_test['is_peak_usage'].values
        test_data.append((X_test, y_test))

    # Evaluate the model
    server_model = server.get_model()
    linear_reg = LinearRegression()
    linear_reg.coef_ = server_model[0]
    linear_reg.intercept_ = server_model[1]
    accuracy_scores = []
    for X_test, y_test in test_data:
        y_pred = linear_reg.predict(X_test)
        y_pred_binary = np.round(y_pred)  # Convert continuous predictions to binary
        accuracy = accuracy_score(y_test, y_pred_binary)
        accuracy_scores.append(accuracy)
    avg_accuracy = np.mean(accuracy_scores)
    return avg_accuracy, server.get_model()

def main():
    # Filenames of the training and testing datasets for each node
    train_filenames = [f'datasets/synthetic_electricity/training/synthetic_household_{i}_train.csv' for i in range(9)]
    test_filenames = [f'datasets/synthetic_electricity/test/synthetic_household_{i}_test.csv' for i in range(9)]
    avg_accuracy, server_model = federated_learning(train_filenames, test_filenames)
    print("Average accuracy on the test set:", avg_accuracy * 100, "%")

    # Select a specific household for plotting
    selected_household = 1
    df_test = pd.read_csv(test_filenames[selected_household])
    X_test = df_test[['time_slot', 'electricity_usage']].values
    y_test = df_test['is_peak_usage'].values

    # Create the LinearRegression model with the aggregated parameters
    linear_reg = LinearRegression()
    linear_reg.coef_ = server_model[0]
    linear_reg.intercept_ = server_model[1]
    # Make predictions
    y_pred = linear_reg.predict(X_test)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(df_test['time_slot'], y_test, label='Actual', marker='o')
    plt.plot(df_test['time_slot'], y_pred, label='Predicted', marker='x')
    plt.title(f'Electricity Usage Prediction for Household {selected_household}')
    plt.xlabel('Time Slot')
    plt.ylabel('Peak Usage Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()