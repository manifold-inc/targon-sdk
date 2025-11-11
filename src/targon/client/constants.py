DEFAULT_BASE_URL = "http://localhost:80"
API_VERSION = "v1"

# API Endpoints
INVENTORY_ENDPOINT = f"/{API_VERSION}/capacity"
DEPLOYMENT_ENDPOINT = f"/{API_VERSION}/serverless"

# Heim Build Service
HEIM_BASE_URL = "https://heim.185.209.179.245.nip.io"
HEIM_BUILD_ENDPOINT = "/build"

# Function Service
FUNC_REG_ENDPOINT = "/" + API_VERSION + "/apps/{app_id}/functions"

# App Service
GET_APP_ENDPOINT = f"/{API_VERSION}/apps"
GET_APP_STATUS_ENDPOINT = f"/{API_VERSION}/apps/{{app_id}}"
LIST_APPS_ENDPOINT = f"/{API_VERSION}/apps"
DELETE_APP_ENDPOINT = f"/{API_VERSION}/apps/{{app_id}}"
LIST_FUNCTIONS_ENDPOINT = f"/{API_VERSION}/apps/{{app_id}}/functions"

# Publish Service
PUBLISH_ENDPOINT = f"/{API_VERSION}/apps/deploy"
