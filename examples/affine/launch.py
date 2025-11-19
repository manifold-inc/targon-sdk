import argparse
import urllib.request
import json
import subprocess


def main(num_models: int, n: int):
    # Fetch weights from affine.io API
    url = "https://dashboard.affine.io/api/weights"
    with urllib.request.urlopen(url) as response:
        resp_json = json.loads(response.read().decode())

    # Example: extract the list of weights from the third column in rows
    # The API returns something like: resp_json['data']['rows']
    affine_model_names = [row[2] for row in resp_json['data']['rows']][:num_models]

    # Call targon run affine_validator.py for each model (concurrently)
    processes = []
    for model_name in affine_model_names:
        cmd = [
            "targon",
            "run",
            "affine_validator.py",
            "--model-name", model_name,
            "--env", "sat",
            "--n", str(n)
        ]
        print(f"Launching: {' '.join(cmd)}")
        processes.append((model_name, cmd, subprocess.Popen(cmd)))

    errors = []
    for model_name, cmd, proc in processes:
        retcode = proc.wait()
        if retcode != 0:
            errors.append((model_name, cmd, retcode))

    if errors:
        first_failed = errors[0]
        raise subprocess.CalledProcessError(first_failed[2], first_failed[1])


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Fetch model names from affine.io API")
    parser.add_argument("--num_models", type=int, default=3, help="Number of models to fetch")
    parser.add_argument("--n", type=int, default=3, help="Number of samples to evaluate")
    args = parser.parse_args()
    num_models = args.num_models
    n = args.n
    main(num_models, n)
