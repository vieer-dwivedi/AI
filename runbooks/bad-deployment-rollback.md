# Runbook: Bad deployment (errors start right after a release)

## Symptoms
- 5xx error rate jumps within minutes of a deployment
- A new exception appears in logs (for example NullPointerException) with the same stack trace on every request
- CPU and memory are normal

## Diagnosis
1. Compare the alarm start time with the latest deployment time.
2. Read the top exception and its stack trace; check it did not exist before the release.
3. Confirm the previous task definition revision was healthy.

## Remediation
1. Roll back the service to the previous task definition revision: `aws ecs update-service --cluster <c> --service <s> --task-definition <family>:<previous>`. Risk: low.
2. Verify 5xx metric returns to baseline within 5 minutes.
3. Block the faulty release in the pipeline and open a ticket with the stack trace for the author.

## Escalation
Page the release owner and engineering manager if the rollback fails.
