DEFAULT_BASE_URL = "http://localhost:8080"
API_VERSION = "v1"
API_VERSION_V2 = "/tha/v2"

# API Endpoints
INVENTORY_ENDPOINT = f"{API_VERSION_V2}/inventory"

# Heim Build Service
HEIM_BASE_URL = "https://api.targon.com"
HEIM_BUILD_ENDPOINT = f"/{API_VERSION}/heim/build"

# Workload Service
WORKLOADS_ENDPOINT = f"{API_VERSION_V2}/workloads"
WORKLOAD_DETAIL_ENDPOINT = f"{API_VERSION_V2}/workloads/{{workload_uid}}"
WORKLOAD_DEPLOY_ENDPOINT = f"{API_VERSION_V2}/workloads/{{workload_uid}}/deploy"
WORKLOAD_STATE_ENDPOINT = f"{API_VERSION_V2}/workloads/{{workload_uid}}/state"
WORKLOAD_EVENTS_ENDPOINT = f"{API_VERSION_V2}/workloads/{{workload_uid}}/events"

# App Service
CREATE_APP_ENDPOINT = f"{API_VERSION_V2}/apps"
GET_APP_ENDPOINT = f"{API_VERSION_V2}/apps/{{app_uid}}"
LIST_APPS_ENDPOINT = f"{API_VERSION_V2}/apps"
DELETE_APP_ENDPOINT = f"{API_VERSION_V2}/apps/{{app_uid}}"

# Logs Service
WORKLOAD_LOGS_ENDPOINT = f"{API_VERSION_V2}/workloads/{{workload_uid}}/logs"
