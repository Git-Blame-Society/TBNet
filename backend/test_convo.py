import convolution

def test_prediction(path):
    prob = convolution.PredictImage(path)
    print(f"TB Probability: {prob:.4f}")
    return { prob }
