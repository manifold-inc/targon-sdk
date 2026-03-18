import asyncio

from targon import Client
from targon.client.serverless import CreateServerlessResponse, ReplicasConfig


async def deploy_container() -> CreateServerlessResponse:
    return await Client.async_serverless.deploy_container(
        name="inspect-container",
        image="nginx:latest",
        port=80,
        replicas=ReplicasConfig(min=0, max=1),
    )


async def inspect_container(workload_uid: str) -> None:
    state = await Client.async_serverless.get_state(workload_uid)
    print(f"\nState for {workload_uid}:")
    print(
        f"  - status={state.status} message={state.message} replicas={state.ready_replicas}/{state.total_replicas}"
    )
    for url in state.urls:
        print(f"  - url[{url.port}]={url.url}")

    events = await Client.async_serverless.get_events(workload_uid, limit=5)
    print(f"\nRecent events for {workload_uid}:")
    for event in events.items:
        message = event.display_message or event.message or event.reason or "-"
        print(
            f"  - {event.created_at} {event.event_type} status={event.new_status or '-'} message={message}"
        )

    print(f"\nRecent logs for {workload_uid}:")
    async for line in Client.async_logs.stream_logs(workload_uid, follow=False):
        print(f"  {line}")


async def main() -> None:
    deployment = await deploy_container()
    print(
        f"Deployed inspect container: {deployment.name} {deployment.uid} {deployment.url}"
    )

    await asyncio.sleep(10)
    await inspect_container(deployment.uid)

    await Client.async_serverless.delete_container(deployment.uid)
    print(f"\nDeleted {deployment.uid}")


if __name__ == "__main__":
    asyncio.run(main())
