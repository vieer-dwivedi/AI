# Runbook: AKS pod CrashLoopBackOff (Azure)

## Symptoms
- Pod status CrashLoopBackOff, restart count increasing
- Azure Monitor alert on container restarts

## Diagnosis
1. `kubectl describe pod` and check the last state reason (OOMKilled, Error).
2. `kubectl logs --previous` for the last crash output.
3. Check config maps and secrets changed recently.

## Remediation
1. Roll back the deployment: `kubectl rollout undo deployment/<name>`. Risk: low.
2. If OOMKilled, raise the memory limit. Risk: medium.
