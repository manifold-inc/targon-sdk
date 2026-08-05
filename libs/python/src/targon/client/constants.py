DEFAULT_BASE_URL = "https://api.targon.com"
API_VERSION_V3 = "/tha/v3"


def org_path(org: str, resource: str) -> str:
    """Build a v3 path for a resource scoped to an organization."""
    if not isinstance(org, str) or not org.strip() or "/" in org:
        raise ValueError("org must be a non-empty organization slug")
    if not isinstance(resource, str) or not resource.strip():
        raise ValueError("resource must be a non-empty path")
    return f"{API_VERSION_V3}/orgs/{org.strip()}/{resource.lstrip('/')}"


# Global resources
INVENTORY_ENDPOINT = f"{API_VERSION_V3}/inventory"
PERSONAL_API_TOKENS_ENDPOINT = f"{API_VERSION_V3}/me/api-tokens"
PERSONAL_API_TOKEN_DETAIL_ENDPOINT = f"{PERSONAL_API_TOKENS_ENDPOINT}/{{token_uid}}"

# Organization resources used by the other v3 clients.
ORGS_ENDPOINT = f"{API_VERSION_V3}/orgs"
ORG_DETAIL_ENDPOINT = f"{ORGS_ENDPOINT}/{{org_slug}}"
ORG_WALLET_ENDPOINT = f"{ORG_DETAIL_ENDPOINT}/wallet"
ORG_CREDITS_ENDPOINT = f"{ORG_DETAIL_ENDPOINT}/credits"
MEMBERS_ENDPOINT = f"{ORG_DETAIL_ENDPOINT}/members"
MEMBER_DETAIL_ENDPOINT = f"{MEMBERS_ENDPOINT}/{{username}}"
SERVICE_TOKENS_ENDPOINT = f"{ORG_DETAIL_ENDPOINT}/tokens"
SERVICE_TOKEN_DETAIL_ENDPOINT = f"{SERVICE_TOKENS_ENDPOINT}/{{token_uid}}"
