from itertools import product
from tqdm import tqdm_notebook
import statsmodels.api as sm
import pandas as pd


# Train many SARIMA models to find the best set of parameters


def sarima_model(data, validation):
    # Set initial values and some bounds
    ps = range(0, 5)
    d = 1
    qs = range(0, 5)
    Ps = range(0, 5)
    D = 1
    Qs = range(0, 5)
    s = 5
    print("initial data")
    print(data.head())
    # Create a list with all possible combinations of parameters
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)

    def optimize_SARIMA(data, parameters_list, d, D, s):
        """
            Return dataframe with parameters and corresponding AIC

            parameters_list - list with (p, q, P, Q) tuples
            d - integration order
            D - seasonal integration order
            s - length of season
        """

        results = []
        best_aic = float('inf')
        i = 0
        r = data.columns[0]
        for param in tqdm_notebook(parameters_list):
            if i > 0:
                break
            print(tqdm_notebook(parameters_list))
            try:
                model = sm.tsa.statespace.SARIMAX(data[r], order=(param[0], d, param[1]),
                                                  seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
            except:
                continue
            i += 1
            aic = model.aic
            # Save best model, AIC and parameters
            if aic < best_aic:
                best_model = model
                best_aic = aic
                best_param = param
            results.append([param, model.aic])

        result_table = pd.DataFrame(results)
        result_table.columns = ['parameters', 'aic']
        # Sort in ascending order, lower AIC is better
        result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

        return result_table

    result_table = optimize_SARIMA(data, parameters_list, d, D, s)

    # Set parameters that give the lowest AIC (Akaike Information Criteria)
    p, q, P, Q = result_table.parameters[0]

    best_model = sm.tsa.statespace.SARIMAX(data.CLOSE, order=(p, d, q),
                                           seasonal_order=(P, D, Q, s)).fit(disp=-1)
    print("SARIMA")
    print(best_model.summary())
    print("len: ", len(validation))
    y_pred = best_model.predict(len(validation))

    import numpy as np
    y_pred = np.array(y_pred.values)
    print("y_pred:\n", y_pred)
    print(validation.values)
    y_true = np.array(validation.CLOSE)
    print(y_true)
    print("y_true shape: ", y_true.shape, " y_pred:", len(y_pred))
    mae = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print("mae: ", mae)

    return 0
