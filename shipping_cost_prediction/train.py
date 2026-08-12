import joblib
from pathlib import Path

from data_preprocessing import load_vendor_invoice_data , prepare_features, split_data

from model_evaluation import train_linear_regression, train_decision_tree, train_random_forest, evaluate_model

def main():
    db_path = "/Users/Raj Gupta/OneDrive/Desktop/datascience/ml project/data/inventory.db"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    #load data
    df = load_vendor_invoice_data(db_path)

    #prepare data
    X,y = prepare_features(df)
    X_train, X_test, Y_train, Y_test = split_data(X,y)
    
    #train models
    lr_model = train_linear_regression(X_train , Y_train)
    dt_model = train_decision_tree(X_train , Y_train)
    rf_model = train_random_forest(X_train, Y_train)

    #Evaluate models
    results = []
    results.append(evaluate_model(lr_model, X_test,Y_test, "linear regression"))
    results.append(evaluate_model(dt_model, X_test,Y_test, "decision tree regressor"))
    results.append(evaluate_model(rf_model, X_test,Y_test, "random forest regressor"))

    # Print model evaluation results
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)

    for result in results:
        print(f"\nModel: {result['model_name']}")
        print(f"MAE:    {result['mae']:.2f}")
        print(f"RMSE:   {result['rmse']:.2f}")
        print(f"R2:     {result['r2']:.4f}")

    #select best models (lowest mae)
    best_model_info = min(results, key=lambda x: x["mae"])
    best_model_name = best_model_info["model_name"]

    best_model = {
        "linear regression" : lr_model,
        "descision tree regressor" : dt_model,
        "random forest regressor" : rf_model
    }[best_model_name]

    #save best model
    model_path = model_dir/"predict_shipping_cost_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    main()
