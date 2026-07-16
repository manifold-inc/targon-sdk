import targon

print("Creating sandbox")
s = targon.Sandbox.create(image="ubuntu", resource=targon.Resources.H100_MEDIUM, keep_alive=True)
print(f"Sandbox created ({s.id})")

response = s.exec('apt update && apt install -y wget', timeout=10)
if response.exit_code != 0:
    print(f"Error: {response.exit_code} {response.result}")
else:
    print(response.result)


response = s.exec('wget https://releases.ubuntu.com/24.04.2/ubuntu-24.04.2-live-server-amd64.iso', timeout=10)
if response.exit_code != 0:
    print(f"Error: {response.exit_code} {response.result}")
else:
    print(response.result)

print("Removing sandbox")
targon.Sandbox.terminate(s)
print("Sandbox removed")
