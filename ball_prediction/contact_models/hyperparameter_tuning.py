import optuna
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV


def bayesian_optimization(
    model_class, training_class, input_data, input_dim, output_dim
):
    param_space = {
        "hidden_neurons": [(32, 256), (32, 256)],
        "learning_rate": (1e-5, 1e-1, "log-uniform"),
        "dropout_rate": (0.0, 0.5),
        "use_layer_norm": [True, False],
    }

    model = model_class(input_dim, output_dim, hidden_neurons=[64, 128])
    trainer = training_class(model, num_epochs=100, learning_rate=0.001)

    bayes_search = BayesSearchCV(
        estimator=trainer, search_spaces=param_space, n_iter=50, cv=5, n_jobs=-1
    )

    bayes_search.fit(input_data)

    best_params = bayes_search.best_params_
    best_score = bayes_search.best_score_

    return best_params, best_score


def random_search(model_class, training_class, input_data, input_dim, output_dim):
    param_dist = {
        "hidden_neurons": [(32, 64, 128, 256), (32, 64, 128, 256)],
        "learning_rate": [0.0001, 0.001, 0.01, 0.1],
        "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "use_layer_norm": [True, False],
    }

    model = model_class(input_dim, output_dim, hidden_neurons=[64, 128])
    trainer = training_class(model, num_epochs=100, learning_rate=0.001)

    random_search = RandomizedSearchCV(
        estimator=trainer, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1
    )

    random_search.fit(input_data)

    best_params = random_search.best_params_
    best_score = random_search.best_score_

    return best_params, best_score


def population_based_search(
    model_class, training_class, input_data, input_dim, output_dim
):
    def objective(trial):
        # Define parameter search space
        hidden_neurons = trial.suggest_categorical(
            "hidden_neurons", [(32, 64), (64, 128), (128, 256)]
        )
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        use_layer_norm = trial.suggest_categorical("use_layer_norm", [True, False])

        model = model_class(input_dim, output_dim, hidden_neurons=hidden_neurons)
        trainer = training_class(model, num_epochs=100, learning_rate=learning_rate)

        # Train the model and get the loss
        loss = trainer.train(input_data)

        return loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_score = study.best_value

    return best_params, best_score
