"""The five candidate classifiers (one per algorithm family) and their initial configurations."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from .config import SEED

# Note: no class_weight="balanced". Re-weighting distorts the probabilities, and the
# probabilities are the HMM emission scores. Ranking within a point is unaffected.
MODEL_SPECS = {
    "gaussian_nb": dict(
        family="Probabilistic", config="var_smoothing=1e-9, standardised inputs",
        objective="Closed-form maximum likelihood of per-class Gaussian feature densities (no iterative loss)",
        build=lambda: make_pipeline(StandardScaler(), GaussianNB())),
    "logistic_regression": dict(
        family="Linear", config="L2, C=1.0, lbfgs, max_iter=1000, standardised inputs",
        objective="Binary cross-entropy (log loss) + L2 penalty, minimised with L-BFGS",
        build=lambda: make_pipeline(StandardScaler(),
                                    LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))),
    "random_forest": dict(
        family="Bagging ensemble", config="150 trees, min_samples_leaf=10, max_features=sqrt",
        objective="Greedy Gini-impurity reduction at each split; trees averaged (no global loss)",
        build=lambda: RandomForestClassifier(n_estimators=150, min_samples_leaf=10, max_features="sqrt",
                                             n_jobs=-1, random_state=SEED)),
    "hist_gradient_boosting": dict(
        family="Boosting", config="lr=0.1, max_iter=300, max_leaf_nodes=31, early stopping",
        objective="Binary log loss, minimised by gradient boosting (Newton steps on histogram trees)",
        build=lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
                                                     early_stopping=True, validation_fraction=0.1,
                                                     random_state=SEED)),
    "mlp": dict(
        family="Neural network", config="hidden=(64,32), relu, adam lr=1e-3, batch=512, early stopping",
        objective="Binary cross-entropy + L2 (alpha=1e-4), minimised with Adam mini-batch SGD",
        build=lambda: make_pipeline(StandardScaler(),
                                    MLPClassifier(hidden_layer_sizes=(64, 32), learning_rate_init=1e-3,
                                                  batch_size=512, max_iter=60, early_stopping=True,
                                                  n_iter_no_change=5, random_state=SEED))),
}
