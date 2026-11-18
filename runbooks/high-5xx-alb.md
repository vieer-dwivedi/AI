# Runbook: High 5xx on Application Load Balancer

## Symptoms
- HTTPCode_Target_5XX_Count above threshold
- Healthy host count dropping

## Diagnosis
1. Identify which target service owns the errors.
2. Check recent deployments, then memory/CPU of the service, then its dependencies (database, cache).
3. Look at the dominant error message in the service logs.

## Remediation
Follow the runbook matching the dominant cause: bad deployment, OOM, or database connections.
