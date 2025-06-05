from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml.entities import Environment

import uuid

# Connect to your workspace
from azure.ai.ml.entities import CodeConfiguration

# Authenticate and create MLClient
credential = DefaultAzureCredential()
subscription_id = ""  # Replace with your Azure subscription ID
resource_group = "  "
workspace_name = "  "  # Replace with your Azure ML workspace name
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Define endpoint name (create or update)
endpoint_name = "diabetes-endpoint-05061"

# Create or update managed online endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    auth_mode="key",  # or "aad"
)

print(f"Creating or updating endpoint '{endpoint_name}'...")
ml_client.begin_create_or_update(endpoint).result()

# Reference registered model by name
model_name = "diabetes-prod-model"  # Make sure this model is registered in your workspace
model = ml_client.models.get(name=model_name, version="1")  # Specify the version if needed, or use latest

# Define code configuration (folder with score.py and the scoring script)
code_config = CodeConfiguration(
    code="./src",            # path to folder with score.py
    scoring_script="score.py"
)


# Define environment (use existing curated environment or custom one)
env = Environment(
    name="diabetes-env",
    description="Environment for diabetes model",
    conda_file="/Users/marlenepostop/MLOps/src/environment.yml",
    image="mcr.microsoft.com/azureml/base:latest"  # base image (can be changed)
)

registered_env = ml_client.environments.create_or_update(env)
print(f"Registered environment: {registered_env.name} v{registered_env.version}")

# Create managed online deployment
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    code_configuration=code_config,
    environment=env,
    instance_type="Standard_DS3_v2",  # recommended SKU
    instance_count=1
)

print(f"Creating or updating deployment '{deployment.name}'...")
ml_client.begin_create_or_update(deployment).result()

print("Deployment complete!")

# Optionally, get the scoring URI
endpoint = ml_client.online_endpoints.get(name=endpoint_name)
print(f"Scoring URI: {endpoint.scoring_uri}")
