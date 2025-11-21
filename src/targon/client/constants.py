DEFAULT_BASE_URL = "https://api.targon.com"
API_VERSION = "v1"

# API Endpoints
INVENTORY_ENDPOINT = f"/{API_VERSION}/capacity"
DEPLOYMENT_ENDPOINT = f"/{API_VERSION}/serverless"

# Heim Build Service
HEIM_BASE_URL = "https://api.targon.com"
HEIM_BUILD_ENDPOINT = f"/{API_VERSION}/heim/build"

# Function Service
FUNC_REG_ENDPOINT = "/" + API_VERSION + "/apps/{app_id}/functions"

# App Service
GET_APP_ENDPOINT = f"/{API_VERSION}/apps"
GET_APP_STATUS_ENDPOINT = f"/{API_VERSION}/apps/{{app_id}}"
LIST_APPS_ENDPOINT = f"/{API_VERSION}/apps"
DELETE_APP_ENDPOINT = f"/{API_VERSION}/apps/{{app_id}}"
LIST_FUNCTIONS_ENDPOINT = f"/{API_VERSION}/apps/{{app_id}}/functions"
GET_FUNCTION_BY_ID_ENDPOINT = f"/{API_VERSION}/functions/{{function_id}}"

# Publish Service
PUBLISH_ENDPOINT = f"/{API_VERSION}/apps/deploy"

# Logs Service
LOGS_ENDPOINT = f"/{API_VERSION}/serverless/logs"