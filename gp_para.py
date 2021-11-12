import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern, RationalQuadratic, \
    ExpSineSquared, DotProduct
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import warnings

warnings.filterwarnings("ignore")


def gp_tuning(X, Y, kernel_length_scales=np.logspace(-7, 3, 11),
              alpha=np.logspace(-7, -2, 6), noise_level_bounds=None,
              noise_level=1e-14,
              additional_kernels=None, starting_length_scale=5,
              return_table=False, write_table=False,
              table_name="bo_2d_models_table.csv", metric="mean_squared_error",
              return_instance=False):
    """
    @param starting_length_scale: starting length scale
    @param noise_level_bounds: tuple bounds for noise level
    @param additional_kernels: kernels besides the ones in the body of this function
    @param return_instance: when true return the gridSearch instance; otherwise, return the best estimator only
    @param metric: metric for deciding the best estimator
    @param noise_level: noise level
    @param X: input data
    @param Y: labels
    @param kernel_length_scales: an array of kernel length scale upperbound
    @param alpha: an array of alpha values
    @param return_table: when true will return a pandas dataframe of training results
    @param write_table: when true will save the table to a csv file
    @param table_name: specify the name of the csv file containing the result table
    @return: depending on input
    """
    # generating kernel choices. Feel free to add more kernels
    kernel_choices = []

    if hasattr(kernel_length_scales, '__iter__'):
        for i in kernel_length_scales:
            kernel_choices.extend([1.0 * RBF(length_scale=starting_length_scale, length_scale_bounds=(0, i)) + WhiteKernel(
                noise_level=noise_level, noise_level_bounds=noise_level_bounds),
                                   1.0 * RBF(length_scale=starting_length_scale, length_scale_bounds=(0, i)) + WhiteKernel(
                                       noise_level=noise_level, noise_level_bounds=noise_level_bounds) + C(),
                                   1.0 * Matern(length_scale=starting_length_scale, length_scale_bounds=(0, i)) + WhiteKernel(
                                       noise_level=noise_level, noise_level_bounds=noise_level_bounds),
                                   1.0 * RationalQuadratic(length_scale=starting_length_scale, length_scale_bounds=(0, i)) + WhiteKernel(
                                       noise_level=noise_level,
                                       noise_level_bounds=noise_level_bounds),
                                   C() * RBF(length_scale=starting_length_scale, length_scale_bounds=(0, i)) + WhiteKernel(
                                       noise_level=noise_level, noise_level_bounds=noise_level_bounds),
                                   DotProduct(sigma_0_bounds=(0, i)) + WhiteKernel(noise_level=noise_level,
                                                                                   noise_level_bounds=noise_level_bounds),
                                   ExpSineSquared(length_scale=starting_length_scale, length_scale_bounds=(0, i)) + WhiteKernel(
                                       noise_level=noise_level, noise_level_bounds=noise_level_bounds)])

    else:
        kernel_choices = [1.0 * RBF(length_scale=starting_length_scale) + WhiteKernel(noise_level=noise_level),
                          1.0 * RBF(length_scale=starting_length_scale) + WhiteKernel(noise_level=noise_level) + C(),
                          1.0 * Matern(length_scale=starting_length_scale) + WhiteKernel(noise_level=noise_level),
                          1.0 * RationalQuadratic(length_scale=starting_length_scale) + WhiteKernel(
                              noise_level=noise_level),
                          C() * RBF(length_scale=starting_length_scale) + WhiteKernel(noise_level=noise_level),
                          DotProduct(sigma_0=starting_length_scale) + WhiteKernel(noise_level=noise_level),
                          ExpSineSquared(length_scale=starting_length_scale) + WhiteKernel(noise_level=noise_level)]

    if additional_kernels:
        kernel_choices.extend(additional_kernels)
    param_space = {'alpha': alpha,
                   'kernel': kernel_choices}
    gp = GaussianProcessRegressor()
    gp_search = GridSearchCV(gp, param_space,
                             scoring={"mean_squared_error": make_scorer(mean_squared_error, greater_is_better=False),
                                      "log_marginal_likelihood": lambda estimator, x,
                                                                        y: estimator.log_marginal_likelihood_value_}
                             , return_train_score=True, refit=metric)
    gp_search.fit(X, Y)

    if write_table:
        res = pd.DataFrame(gp_search.cv_results_)
        col_reorder_select = ['param_kernel',
                              'mean_train_log_marginal_likelihood', 'mean_train_mean_squared_error',
                              'mean_test_log_marginal_likelihood', 'mean_test_mean_squared_error',
                              'param_alpha', 'params',
                              'std_train_mean_squared_error', 'std_test_log_marginal_likelihood',
                              'std_test_mean_squared_error', 'rank_test_mean_squared_error',
                              'rank_test_log_marginal_likelihood'
                              ]
        res = res[col_reorder_select]
        res.sort_values(by=["mean_train_log_marginal_likelihood", 'mean_test_log_marginal_likelihood',
                            'mean_train_mean_squared_error', 'mean_test_mean_squared_error']
                        , ascending=False, inplace=True)
        res.to_csv(table_name)

        if return_table:
            if return_instance:
                return res, gp_search
            return res, gp_search.best_estimator_

    return gp_search.best_estimator_


def model_performance(kernel, X, Y, txt):
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
    models = cross_validate(gp, X, Y, return_estimator=True, cv=len(Y), scoring="neg_mean_squared_error",
                            return_train_score=True)
    for i in models["estimator"]:
        txt.write(f"Learned kernel: {i.kernel_}\n")
    txt.write(f"\ntest scores for each model: {models['test_score']} \n\n")
    txt.write(f"negative mean squared error: {models['train_score']} \n\n")
    txt.write(f"log marginal likelihood: {[i.log_marginal_likelihood() for i in models['estimator']]} \n\n")
    return models
