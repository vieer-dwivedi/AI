# Runbook: High CPU on EC2

## Symptoms
- CPUUtilization above 85% for 15 minutes

## Diagnosis
1. Check top processes through SSM Session Manager.
2. Look for runaway cron jobs or traffic spikes.

## Remediation
1. Restart the runaway process. Risk: low.
2. Scale out the Auto Scaling group. Risk: low.
