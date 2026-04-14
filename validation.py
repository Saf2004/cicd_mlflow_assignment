import sys
import mlflow

THRESHOLD = 0.95

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking run: {run_id}")

run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0.0)

print(f"Accuracy: {accuracy:.4f}")

if accuracy < THRESHOLD:
    print(f"FAILED: accuracy {accuracy:.4f} < threshold {THRESHOLD}")
    sys.exit(1)

print(f"PASSED: accuracy {accuracy:.4f} >= threshold {THRESHOLD}")
