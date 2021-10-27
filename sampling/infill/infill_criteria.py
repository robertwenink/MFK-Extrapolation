"""

TODO ego-ei voor incraesing mf? nieuw ontwikkelen, of blijft hetzelfde? literatuur?
 i.e. if we have sample uncertainty, we have a probability distribution of the value of Y_min as well + possibly different ponts.
"""


# https://github.com/mid2SUPAERO/multi-fi-optimization/blob/master/MFK_tutorial.ipynb
import scipy  as sci
def EI_function(model,x):
    x = np.atleast_2d(x)
    Y_min = np.min(model.training_points[None][0][1])
    y_pred = model.predict_values(x)
    mse = modem.predict_variances(x)
    y_pred = np.atleast_2d(y_pred)
    sigma_y = np.sqrt(mse)
    y_normed = (Y_min - y_pred[:,0])/sigma_y
    " deze lijn gaat het om "
    EI = (Y_min-y_pred[:,0])*sci.stats.norm.cdf(y_normed)+sigma_y*sci.stats.norm.pdf(y_normed)
    return -EI