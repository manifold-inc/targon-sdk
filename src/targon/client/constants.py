DEFAULT_BASE_URL = "http://localhost:8080"
API_VERSION = "v1"
API_VERSION_V2 = "/tha/v2"

# API Endpoints
INVENTORY_ENDPOINT = f"{API_VERSION_V2}/inventory"
SERVERLESS_ENDPOINT = f"/{API_VERSION}/serverless"

# Heim Build Service
HEIM_BASE_URL = "https://api.targon.com"
HEIM_BUILD_ENDPOINT = f"/{API_VERSION}/heim/build"

# Function Service
FUNC_REG_ENDPOINT = "/" + API_VERSION + "/apps/{app_id}/functions"

# App Service
CREATE_APP_ENDPOINT = f"{API_VERSION_V2}/apps"
GET_APP_ENDPOINT = f"{API_VERSION_V2}/apps/{{app_uid}}"
GET_APP_STATUS_ENDPOINT = f"/{API_VERSION}/apps/{{app_id}}"
LIST_APPS_ENDPOINT = f"{API_VERSION_V2}/apps"
DELETE_APP_ENDPOINT = f"{API_VERSION_V2}/apps/{{app_uid}}"
LIST_FUNCTIONS_ENDPOINT = f"/{API_VERSION}/apps/{{app_id}}/functions"
GET_FUNCTION_BY_ID_ENDPOINT = f"/{API_VERSION}/functions/{{function_id}}"

# Publish Service
PUBLISH_ENDPOINT = f"/{API_VERSION}/apps/deploy"

# Logs Service
LOGS_ENDPOINT = f"/{API_VERSION}/serverless/logs"
