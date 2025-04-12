import numpy as np
from scipy.optimize import minimize


class PortfolioOptimizer:
    def __init__(self, returns, mean_returns, cov_matrix, target_type, target_return=None):
        self.returns = returns
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.target_type = target_type
        self.target_return = target_return

    def num_of_assets(self, mean_returns):
        return len(mean_returns)

    def optimize(self):
        num_assets = self.num_of_assets(self.mean_returns)
        init_guess = [1. / num_assets] * num_assets
        bounds = [(0.0, 1.0)] * num_assets
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        if self.target_type == "Max Sharpe Ratio":
            def objective(weights):
                ret = np.dot(weights, self.mean_returns)
                vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                return -ret / vol

        elif self.target_type == "Min Volatility":
            def objective(weights):
                return np.dot(weights.T, np.dot(self.cov_matrix, weights))

        elif self.target_type == "Target Return":
            def objective(weights):
                return np.dot(weights.T, np.dot(self.cov_matrix, weights))
            constraints.append({'type': 'eq', 'fun': lambda x: np.dot(x, self.mean_returns) - self.target_return})

        result = minimize(objective, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
            return weights
        else:
            raise ValueError("Optimization failed")
