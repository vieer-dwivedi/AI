# Runbook: RDS / PostgreSQL connection exhaustion

## Symptoms
- DatabaseConnections metric flat at the maximum (for example 100)
- Postgres log: "remaining connection slots are reserved for non-replication superuser connections"
- App log: "HikariPool-1 - Connection is not available, request timed out"
- HTTP 500 errors from services that use the database

## Diagnosis
1. Check DatabaseConnections history and when it started rising.
2. In the Postgres log, count connections by application_name and user to find the heavy client.
3. Check for new jobs, cron tasks or releases started just before the rise.
4. CPU and memory are often normal, so do not rule out a connection problem because CPU is fine.

## Remediation
1. Stop or pause the client that is opening too many connections (for example a batch or reporting job). Risk: low.
2. Terminate idle connections of that client with pg_terminate_backend filtered by application_name. Risk: medium.
3. Add connection pooling (RDS Proxy or PgBouncer) and cap the pool size of the offending job.
4. Only as a last resort raise max_connections or scale the instance class (needs a reboot). Risk: high.

## Escalation
Page the database on-call if connections do not drop within 10 minutes of stopping the client.
