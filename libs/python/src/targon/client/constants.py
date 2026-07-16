DEFAULT_BASE_URL = "https://api.targon.com"
API_VERSION_V2 = "/tha/v2"

# Inventory
INVENTORY_ENDPOINT = f"{API_VERSION_V2}/inventory"

# Workloads
WORKLOADS_ENDPOINT = f"{API_VERSION_V2}/workloads"
WORKLOAD_DETAIL_ENDPOINT = f"{API_VERSION_V2}/workloads/{{workload_uid}}"
WORKLOAD_DEPLOY_ENDPOINT = f"{API_VERSION_V2}/workloads/{{workload_uid}}/deploy"
WORKLOAD_STATE_ENDPOINT = f"{API_VERSION_V2}/workloads/{{workload_uid}}/state"
WORKLOAD_EVENTS_ENDPOINT = f"{API_VERSION_V2}/workloads/{{workload_uid}}/events"
WORKLOAD_LOGS_ENDPOINT = f"{API_VERSION_V2}/workloads/{{workload_uid}}/logs"
WORKLOAD_EXEC_ENDPOINT = f"{API_VERSION_V2}/workloads/{{workload_uid}}/exec"
WORKLOAD_VERIFY_ENDPOINT = f"{API_VERSION_V2}/workloads/verify"
WORKLOAD_VOLUME_ENDPOINT = (
    f"{API_VERSION_V2}/workloads/{{workload_uid}}/volumes/{{volume_uid}}"
)
WORKLOAD_SSH_KEY_ENDPOINT = (
    f"{API_VERSION_V2}/workloads/{{workload_uid}}/ssh-keys/{{ssh_key_uid}}"
)

# Projects
PROJECTS_ENDPOINT = f"{API_VERSION_V2}/projects"
PROJECT_DETAIL_ENDPOINT = f"{API_VERSION_V2}/projects/{{project_uid}}"

# Volumes
VOLUMES_ENDPOINT = f"{API_VERSION_V2}/volumes"
VOLUME_DETAIL_ENDPOINT = f"{API_VERSION_V2}/volumes/{{volume_uid}}"
VOLUME_STATE_ENDPOINT = f"{API_VERSION_V2}/volumes/{{volume_uid}}/state"
VOLUME_EVENTS_ENDPOINT = f"{API_VERSION_V2}/volumes/{{volume_uid}}/events"
VOLUME_DELETE_DEPLOYMENT_ENDPOINT = f"{API_VERSION_V2}/volumes/{{volume_uid}}/delete"

# SSH Keys
SSH_KEYS_ENDPOINT = f"{API_VERSION_V2}/ssh-keys"
SSH_KEY_DETAIL_ENDPOINT = f"{API_VERSION_V2}/ssh-keys/{{ssh_key_uid}}"

# User
USER_WALLET_ENDPOINT = f"{API_VERSION_V2}/me/wallet"
USER_CREDITS_ENDPOINT = f"{API_VERSION_V2}/me/credits"
USER_API_KEYS_ENDPOINT = f"{API_VERSION_V2}/me/api-keys"
USER_API_KEY_DETAIL_ENDPOINT = f"{API_VERSION_V2}/me/api-keys/{{key_uid}}"
USER_API_KEY_ROTATE_ENDPOINT = f"{API_VERSION_V2}/me/api-keys/{{key_uid}}:roll"
