import score
import json
import joblib



# Initialize model (like Azure does)
score.init()

# Load sample data from JSON file
with open("/Users/marlenepostop/MLOps/sample_input.json", "r") as f:
    input_data = f.read()

# Call run() with sample data
output = score.run(input_data)

print("Prediction output:", output)
