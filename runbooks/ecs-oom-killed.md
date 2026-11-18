# Runbook: ECS tasks OOMKilled / OutOfMemoryError

## Symptoms
- ECS tasks stop with exit code 137 or "OutOfMemoryError: Container killed due to memory usage"
- ECS MemoryUtilization near 100%, runningCount below desiredCount
- Logs show "java.lang.OutOfMemoryError: Java heap space" or GC overhead warnings
- ALB returns 502/503/5xx while tasks restart

## Diagnosis
1. Check ECS MemoryUtilization trend; note when it started climbing.
2. Compare with the last deployment time. A climb right after a release means the release caused it.
3. Read the deployment description for cache, batch size or heap changes.
4. Check the task stopped reasons in service events.

## Remediation
1. If a deployment happened shortly before the climb: roll back the ECS service to the previous task definition revision (`aws ecs update-service --task-definition <family>:<previous>`). Risk: low.
2. If no recent deployment: raise task memory (new task definition revision with higher memory) and JVM heap together. Risk: medium.
3. Scale out desiredCount temporarily to absorb traffic while tasks recover.
4. After recovery, open a ticket to profile heap usage before re-releasing.

## Escalation
Page the owning service team if rollback does not restore runningCount within 10 minutes.
