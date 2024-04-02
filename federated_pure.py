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
        self.model_updates = []
    
    def aggregate_updates(self, updates):
        avg_coef = np.mean([update[0] for update in updates], axis=0)
        avg_intercept = np.mean([update[1] for update in updates], axis=0)
        self.model.coef_ = avg_coef
        self.model.intercept_ = avg_intercept
    
    def get_model(self):
        return self.model.coef_, self.model.intercept_

def federated(filename):

    # Load the CSV file for a single node
    df = pd.read_csv(filename, delimiter=';')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S')

    df.set_index('datetime', inplace=True)

    # Preprocess the data (this is just an example)
    X = df[['diff']].values
    y = df['meter.reading'].values

    # Create a federated client with the preprocessed data
    client = FederatedClient(LinearRegression(), (X, y))

    # Simulate federated learning (this is a simplified example)
    server = FederatedServer(LinearRegression())

    for round in range(10):
        client.train()
        update = client.get_model_update()
        server.aggregate_updates([update])
        client.set_model(server.get_model())

    # The client now has an updated model
    #print(client.model.coef_)
    return client.model.coef_

def main():
    # Run federated learning on a single node
    #Reiterate through node1.csv, node2.csv, node3.csv, node4.csv
    print(federated('datasets/node1.csv'))
    print(federated('datasets/node2.csv'))
    print(federated('datasets/node3.csv'))
    print(federated('datasets/node4.csv'))

if __name__ == "__main__":
    main()

