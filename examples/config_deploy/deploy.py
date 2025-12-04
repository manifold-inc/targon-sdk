import asyncio
import time
import targon
from targon import Client
from targon.client.serverless import CreateServerlessResponse, ReplicasConfig


async def deploy_container(i: int) -> CreateServerlessResponse:
    deployment: CreateServerlessResponse = await targon.Client.async_serverless.deploy_container(
            name=f"container-{i}",
            image="nginx:latest",
            command=["sh", "-c"],
            args=["nginx -g"],
            port=80,
            replicas=ReplicasConfig(
                min=0,
                max=1
            )
    )
    return deployment

async def main():
    tasks = [deploy_container(i) for i in range(3)]
    deployed = await asyncio.gather(*tasks) # Deploy all containers concurrently
    for deployment in deployed:
        print(f"Deployed container {deployment.name}: {deployment.uid} {deployment.url}")

    print(f"\nSleeping 5s")
    time.sleep(10)
    
    print("\nDeleting these deployments.")
    for deployment in deployed:
        _ = await Client.async_serverless.delete_container(deployment.uid)
        print(f"  - Deleted {deployment.uid}")

    all_deployments = await Client.async_serverless.list_container()
    print(f"\nTotal Active deployments: {len(all_deployments)}")
    for deployment in all_deployments:
        print(f"  - {deployment.name} ({deployment.uid})")

if __name__ == "__main__":
    asyncio.run(main())