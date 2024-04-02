import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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

def federated_learning(filenames):
    # Create a federated server with a linear regression model
    server = FederatedServer(LinearRegression())
    
    # Create a federated client for each node and store them in a list
    clients = []
    for filename in filenames:
        df = pd.read_csv(filename, delimiter=';')
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S')
        df.set_index('datetime', inplace=True)
        X = df[['diff']].values
        y = df['meter.reading'].values
        client = FederatedClient(LinearRegression(), (X, y))
        clients.append(client)
    
    # Perform one round of federated learning with sum aggregation
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
    print(f"    Final aggregated weights by Node 4: {server.model.coef_}, intercept: {server.model.intercept_}\n")
    
    # Return the final aggregated model from the server
    return server.get_model()



def main():
    # Filenames of the datasets for each node
    filenames = ['datasets/node1.csv', 'datasets/node2.csv', 'datasets/node3.csv', 'datasets/node4.csv']
    final_model_params = federated_learning(filenames)
    print("Final aggregated model coefficients:", final_model_params[0])
    print("Final aggregated model intercept:", final_model_params[1])

if __name__ == "__main__":
    main()
