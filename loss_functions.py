import numpy as np

def cross_entropy(predicted_output, desired_output):
    samples = len(predicted_output)
    predicted_output_clipped = np.clip(predicted_output, 1e-7, 1-1e-7)
    
    #This occurs when the targets are scalar form
    #i.e the targets are [0, 1]
    #instead of [[1, 0], [0, 1]]
    if len(desired_output.shape) == 1:
        correct_confidences = predicted_output_clipped[range(samples), desired_output]
    
    elif len(desired_output.shape) == 2:
        correct_confidences = np.sum(predicted_output_clipped * desired_output, axis=1)
    
    sample_losses = -np.log(correct_confidences)
    data_loss = np.mean(sample_losses)
    return data_loss
    