import time
import warnings
import logging
import mlflow
import mlflow.sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data / 255.0, mnist.target

X_sampled, _, y_sampled, _ = train_test_split(
    X, y, 
    train_size=8000, 
    stratify=y, 
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_sampled, y_sampled, 
    test_size=0.2, 
    stratify=y_sampled, 
    random_state=42
)

def train_model(hidden_layer_sizes, learning_rate_init, epochs):
    mlflow.set_tracking_uri("sqlite:///my_assignment.db")
    mlflow.set_experiment("MNIST_MLP_Experiments")
    
    run_name = f"hl_{hidden_layer_sizes}_lr_{learning_rate_init}_ep_{epochs}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"Training with hidden_layers={hidden_layer_sizes}, lr={learning_rate_init}, epochs={epochs}...")
        
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, 
            learning_rate_init=learning_rate_init,
            max_iter=epochs,
            random_state=42
        )
        
        start_time = time.time()
        mlp.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        train_acc = mlp.score(X_train, y_train)
        val_acc = mlp.score(X_val, y_val)
        train_loss = mlp.loss_curve_[-1] if hasattr(mlp, 'loss_curve_') else 0.0
        
        mlflow.log_param("hidden_layer_sizes", hidden_layer_sizes)
        mlflow.log_param("learning_rate_init", learning_rate_init)
        mlflow.log_param("epochs", epochs)
        
        mlflow.log_metric("train_acc", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("training_time", training_time)
        
        if hasattr(mlp, 'loss_curve_'):
            for epoch, loss in enumerate(mlp.loss_curve_):
                mlflow.log_metric("train_loss_curve", loss, step=epoch)
        
        mlflow.sklearn.log_model(
            mlp, 
            "model", 
            skops_trusted_types=['sklearn.neural_network._stochastic_optimizers.AdamOptimizer']
        )
        
        print(f"Run completed. Val Accuracy: {val_acc:.4f}\n")

if __name__ == "__main__":
    experiments = [
        ((20,), 0.001, 15),
        ((20,), 0.01, 25),
        ((30,), 0.001, 15),
        ((30,), 0.01, 25),
        ((20, 10), 0.001, 15),
        ((20, 10), 0.01, 25)
    ]
    
    for hidden_layers, lr, epochs in experiments:
        train_model(hidden_layers, lr, epochs)
    
    print("All experiments completed.")
