"""
Quantum-Inspired Evolutionary Algorithm (QIEA) for Feature Selection.

Implements a population of quantum-inspired individuals represented by qubit
amplitudes. Each generation observes binary solutions, evaluates fitness via
cross-validated AUC-ROC, and updates qubits using a quantum rotation gate.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


class QIEAFeatureSelector:
    """
    Quantum-Inspired Evolutionary Algorithm for Feature Selection.

    Parameters
    ----------
    n_features : int
        Total number of features (d).
    pop_size : int
        Population size (m). Default 20.
    n_generations : int
        Maximum generations (T). Default 50.
    delta_theta : float
        Rotation gate step size. Default 0.02 * pi.
    lam : float
        Parsimony penalty weight. Default 0.1.
    fitness_clf : sklearn classifier or None
        Classifier for fitness evaluation. If None, uses
        RandomForestClassifier(n_estimators=50, max_depth=10).
    cv_folds : int
        Number of stratified CV folds for fitness. Default 5.
    max_fitness_samples : int or None
        If set, subsample training data to this many rows for fitness
        evaluation to speed up the algorithm.
    random_state : int
        Random seed. Default 42.
    verbose : bool
        Print progress messages. Default True.
    """

    def __init__(
        self,
        n_features,
        pop_size=20,
        n_generations=50,
        delta_theta=None,
        lam=0.1,
        fitness_clf=None,
        cv_folds=5,
        max_fitness_samples=20000,
        random_state=42,
        verbose=True,
    ):
        self.n_features = n_features
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.delta_theta = delta_theta if delta_theta is not None else 0.02 * np.pi
        self.lam = lam
        self.cv_folds = cv_folds
        self.max_fitness_samples = max_fitness_samples
        self.random_state = random_state
        self.verbose = verbose

        if fitness_clf is None:
            self.fitness_clf = RandomForestClassifier(
                n_estimators=50, max_depth=10, random_state=random_state, n_jobs=-1
            )
        else:
            self.fitness_clf = fitness_clf

        # Will be set during fit
        self.best_mask_ = None
        self.best_fitness_ = -np.inf
        self.convergence_history_ = []
        self.mean_fitness_history_ = []

    def _initialize_population(self):
        """Initialize qubit population with equal superposition."""
        # Shape: (pop_size, 2, n_features) -- row 0 = alpha, row 1 = beta
        population = np.full(
            (self.pop_size, 2, self.n_features), 1.0 / np.sqrt(2.0)
        )
        return population

    def _observe(self, individual):
        """
        Observe a binary solution from a qubit individual.

        Parameters
        ----------
        individual : ndarray, shape (2, n_features)
            Qubit amplitudes [alpha; beta].

        Returns
        -------
        mask : ndarray, shape (n_features,)
            Binary feature mask.
        """
        beta_sq = individual[1, :] ** 2
        r = self.rng_.random(self.n_features)
        mask = (r < beta_sq).astype(np.int32)

        # Ensure at least one feature is selected
        if mask.sum() == 0:
            mask[self.rng_.integers(0, self.n_features)] = 1

        return mask

    def _evaluate_fitness(self, mask, X_train, y_train):
        """
        Evaluate fitness of a binary feature mask.

        fitness = mean(AUC-ROC across CV folds) - lambda * (n_selected / n_features)
        """
        selected = np.where(mask == 1)[0]
        if len(selected) == 0:
            return -np.inf

        X_sel = X_train[:, selected]

        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        try:
            scores = cross_val_score(
                self.fitness_clf,
                X_sel,
                y_train,
                cv=skf,
                scoring="roc_auc",
                n_jobs=-1,
            )
            auc = scores.mean()
        except Exception:
            return -np.inf

        parsimony = self.lam * (mask.sum() / self.n_features)
        return auc - parsimony

    def _update_qubits(self, individual, current_mask, best_mask):
        """
        Update qubit amplitudes using the quantum rotation gate.

        Rotation angle: theta = sign(alpha_j * beta_j) * delta_theta * (best_j - current_j)

        Positive theta increases beta (probability of selecting feature).
        The sign of alpha*beta determines the rotation direction needed
        to move toward the best solution in the current quadrant.
        """
        alpha = individual[0, :]
        beta = individual[1, :]

        # (best_j - current_j): +1 means "should select", -1 means "should deselect"
        delta = best_mask - current_mask
        ab_product = alpha * beta

        # Rotation direction: sign(alpha*beta) * (best - current)
        sign = np.zeros(self.n_features)
        needs_rotation = delta != 0

        # When alpha*beta > 0: rotate in direction of delta
        pos_ab = needs_rotation & (ab_product > 0)
        sign[pos_ab] = np.sign(delta[pos_ab])

        # When alpha*beta < 0: rotate opposite to delta (different quadrant)
        neg_ab = needs_rotation & (ab_product < 0)
        sign[neg_ab] = -np.sign(delta[neg_ab])

        # When alpha ~= 0 or beta ~= 0: pick random direction
        random_signs = self.rng_.choice([-1, 1], size=self.n_features)
        degenerate = needs_rotation & ((np.abs(alpha) < 1e-10) | (np.abs(beta) < 1e-10))
        sign[degenerate] = random_signs[degenerate]

        # Apply rotation gate
        theta = sign * self.delta_theta
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        new_alpha = cos_t * alpha - sin_t * beta
        new_beta = sin_t * alpha + cos_t * beta

        individual[0, :] = new_alpha
        individual[1, :] = new_beta

        return individual

    def fit(self, X_train, y_train):
        """
        Run the QIEA feature selection algorithm.

        Parameters
        ----------
        X_train : ndarray, shape (n_samples, n_features)
        y_train : ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        self.rng_ = np.random.default_rng(self.random_state)

        # Subsample for fitness evaluation if needed
        if self.max_fitness_samples and len(X_train) > self.max_fitness_samples:
            idx = self.rng_.choice(
                len(X_train), self.max_fitness_samples, replace=False
            )
            X_fit = X_train[idx]
            y_fit = y_train[idx]
            if self.verbose:
                print(
                    f"Subsampled {len(X_train)} -> {self.max_fitness_samples} "
                    f"for fitness evaluation"
                )
        else:
            X_fit = X_train
            y_fit = y_train

        population = self._initialize_population()
        self.best_fitness_ = -np.inf
        self.best_mask_ = None
        self.convergence_history_ = []
        self.mean_fitness_history_ = []

        # Store per-individual current observed masks and personal bests
        current_masks = [None] * self.pop_size
        individual_best_masks = [None] * self.pop_size
        individual_best_fitness = np.full(self.pop_size, -np.inf)

        for gen in range(self.n_generations):
            gen_fitnesses = []

            for i in range(self.pop_size):
                # Observe binary solution
                mask = self._observe(population[i])
                current_masks[i] = mask

                # Evaluate fitness
                fitness = self._evaluate_fitness(mask, X_fit, y_fit)
                gen_fitnesses.append(fitness)

                # Update individual personal best
                if fitness > individual_best_fitness[i]:
                    individual_best_fitness[i] = fitness
                    individual_best_masks[i] = mask.copy()

                # Update global best
                if fitness > self.best_fitness_:
                    self.best_fitness_ = fitness
                    self.best_mask_ = mask.copy()

            # Update qubits: rotate each individual toward the global best
            for i in range(self.pop_size):
                population[i] = self._update_qubits(
                    population[i],
                    current_masks[i],
                    self.best_mask_,
                )

            self.convergence_history_.append(self.best_fitness_)
            self.mean_fitness_history_.append(np.mean(gen_fitnesses))

            if self.verbose and (gen + 1) % 5 == 0:
                n_sel = self.best_mask_.sum() if self.best_mask_ is not None else 0
                print(
                    f"Generation {gen + 1}/{self.n_generations} | "
                    f"Best Fitness: {self.best_fitness_:.4f} | "
                    f"Mean Fitness: {np.mean(gen_fitnesses):.4f} | "
                    f"Features: {n_sel}"
                )

        if self.verbose:
            print(
                f"\nQIEA Complete: Best Fitness = {self.best_fitness_:.4f}, "
                f"Features Selected = {self.best_mask_.sum()}/{self.n_features}"
            )

        return self

    def transform(self, X):
        """Apply the best feature mask to select columns."""
        if self.best_mask_ is None:
            raise RuntimeError("Must call fit() before transform().")
        return X[:, self.best_mask_ == 1]

    def fit_transform(self, X_train, y_train):
        """Fit and transform in one step."""
        self.fit(X_train, y_train)
        return self.transform(X_train)

    def get_selected_indices(self):
        """Return indices of selected features."""
        if self.best_mask_ is None:
            raise RuntimeError("Must call fit() before get_selected_indices().")
        return np.where(self.best_mask_ == 1)[0]

    def get_support(self):
        """Return boolean mask of selected features."""
        if self.best_mask_ is None:
            raise RuntimeError("Must call fit() before get_support().")
        return self.best_mask_ == 1
