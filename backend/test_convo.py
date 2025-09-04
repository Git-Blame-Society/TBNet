import convolution

prob, label = convolution.PredictImage("./dataset/Testing Dataset/Data/Normal-669.png")
print(f"TB Probability: {prob:.4f}, Predicted Label: {label}")
