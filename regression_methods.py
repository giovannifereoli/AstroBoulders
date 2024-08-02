from GravModels.utils.Regression import *

# Example usage:
regression = Regression()
regression.load_data("Project/filtered_state_EKF_CR3BP.npy")

# ANN training
regression.train_ann(num_epochs=200000)
regression.save_ann_model("Project/simple_nn_model.pth")
regression.plot_ann_training_loss("Project/TrainingLoss.pdf")
regression.plot_ann_prediction_errors("Project/PredictionError.pdf")

# RLS training
regression.rls_regression(lambda_=0.99, delta=1.0, num_epochs=100)
regression.plot_rls_training_loss("Project/RLSTrainingLoss.pdf")

# ELM training
regression.train_elm(hidden_units=10)
regression.plot_elm_training_loss("Project/ELMTrainingLoss.pdf")

# Compute Jacobian
input_data = [1.0, 0.0, 0.0, 0.0, 0.5, 0.0]  # Example input
jacobian = regression.compute_jacobian(input_data)
print("Jacobian Matrix:")
print(jacobian)
