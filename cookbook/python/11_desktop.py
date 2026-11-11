"""Check desktop availability and mint a short-lived access ticket."""

from _box import open_client, sandbox

client = open_client()
try:
    with sandbox(client, "desk") as box:
        desktop = box.get_desktop()
        print(
            "desktop available=%s port=%s listening=%s"
            % (desktop.available, desktop.port, desktop.listening)
        )
        if desktop.ws_url:
            print("desktop ws url:", desktop.ws_url)

        ticket = box.mint_access_ticket(ttl_sec=60)
        # ticket.ticket is a secret. Print only when it expires.
        print("access ticket expires at", ticket.expires_at)
finally:
    client.close()
