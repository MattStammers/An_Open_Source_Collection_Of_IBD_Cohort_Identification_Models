"""

Sample_size_calculation.py

This sample size estimator uses the formula proposed by Pate and Riley for medical models. 

Assumed shrinkage of 0.9 is used here. 

Reducing the shrinkage further for this project may be appropriate but doesn't have a dramatic effect on the final sample size.

"""

import math

# --------------------------------------------------------------------------- #
# Sample Size Calculation                                                     #
# --------------------------------------------------------------------------- #

def calculate_sample_size(r2, num_predictors, shrinkage, prevalence):
    """
    Calculate the required sample size for a prediction model.

    Parameters:
    r2 (float): Expected Cox-Snell R-squared of the new model
    num_predictors (int): Number of candidate predictor parameters
    shrinkage (float): Desired shrinkage factor
    prevalence (float): Overall outcome prevalence

    Returns:
    float: Required sample size
    """
    # Calculate the log term
    log_term = math.log(1 - r2)

    # Calculate the effective sample size before adjusting for prevalence
    effective_sample_size = num_predictors / (shrinkage * -log_term)

    # Adjust for prevalence
    sample_size = effective_sample_size / (prevalence * (1 - prevalence))

    return sample_size

# Parameters
r2 = 0.05  # Expected R-squared
num_predictors = 4  # Number of candidate predictors
shrinkage = 0.9  # Desired shrinkage
prevalence = 0.8  # Outcome prevalence

# Calculate the sample size
required_sample_size = calculate_sample_size(r2, num_predictors, shrinkage, prevalence)
print(
    f"The sample size should be at least {round(required_sample_size,1)} to guarantee sufficient power for the regression classifier in this study"
)