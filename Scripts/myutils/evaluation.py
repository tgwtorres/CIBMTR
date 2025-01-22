def eval_linear_model(X_test,y_test,mod):
    y_pred = mod.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = np.round(mae,4)
    mse = np.round(mse,4)
    r2 = np.round(r2,4)
    
    return {
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "R-squared": r2
    }

def pred_val(X_test,mod):
    y_pred = mod.predict(X_test)
    y_pred_rounded = np.round(y_pred, 4)

    return y_pred_rounded

def plot_scatter(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
    plt.xlabel('Actual y_test')
    plt.ylabel('Predicted y')
    plt.title('Scatter Plot of Actual vs. Predicted Values')
    plt.grid(True)
    plt.show()
