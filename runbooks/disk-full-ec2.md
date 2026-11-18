# Runbook: EC2 disk full (No space left on device)

## Symptoms
- CWAgent disk_used_percent at or near 100%
- Logs: "OSError: [Errno 28] No space left on device" or "filesystem is full"
- Application writes fail, requests error out

## Diagnosis
1. Check disk_used_percent trend and when growth started.
2. Look for a recent config change, for example DEBUG log level enabled.
3. Find the largest files with `du -xh / | sort -h | tail` (usually /var/log).

## Remediation
1. Reduce log level back to INFO in the app config. Risk: low.
2. Truncate the oversized log file (`: > /var/log/app/debug.log`) and compress or delete rotated logs. Risk: low.
3. Configure logrotate with a size limit for the application logs.
4. If still full, extend the EBS volume (`aws ec2 modify-volume`) and grow the filesystem. Risk: medium.

## Escalation
Escalate to platform team if the volume cannot be extended online.
