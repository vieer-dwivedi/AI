# =====================================================================
# sample_data/generate_sample_data.py
# In simple words: this script CREATES fake-but-realistic AWS data so you
# can test the agent without a real AWS account.
#
# It makes 4 incident "scenarios". Each one is a JSON file that copies the
# real shape of AWS API answers:
#   - CloudWatch Logs   (filter_log_events  -> events with timestamp/message)
#   - CloudWatch Alarms (describe_alarms    -> MetricAlarms)
#   - CloudWatch Metrics(get_metric_data    -> timestamps + values)
#   - ECS / RDS / EC2   (describe_services / describe_db_instances / ...)
#
# Run it any time:   python sample_data/generate_sample_data.py
# It is deterministic (fixed random seed) so tests are stable.
# =====================================================================
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "scenarios"
ACCOUNT = "123456789012"
REGION = "us-east-1"

# The moment every alarm fires in the sample world: 2026-11-18 09:00 UTC.
ANCHOR = datetime(2026, 11, 18, 9, 0, 0, tzinfo=timezone.utc)
ANCHOR_MS = int(ANCHOR.timestamp() * 1000)

rnd = random.Random(7)


# ----------------------------- small helpers -------------------------
def ms(minutes_before: float) -> int:
    """Epoch-milliseconds for 'N minutes before the alarm fired'."""
    return ANCHOR_MS - int(minutes_before * 60_000)


def iso(t_ms: int) -> str:
    """Epoch-ms -> '2026-11-18T08:55:00.123Z'."""
    d = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"


def ev(stream: str, t_ms: int, message: str) -> dict:
    """One CloudWatch log event (same keys as boto3 filter_log_events)."""
    return {
        "logStreamName": stream,
        "timestamp": t_ms,
        "message": message,
        "ingestionTime": t_ms + rnd.randint(200, 900),
        "eventId": f"{rnd.getrandbits(186):056d}"[:56],
    }


def app_json(level: str, t_ms: int, service: str, msg: str, **extra) -> str:
    """A JSON-formatted application log line (very common in AWS)."""
    return json.dumps({"time": iso(t_ms), "level": level, "service": service,
                       "message": msg, **extra})


def noise(group_streams, service, minutes, per_min=2):
    """Healthy background traffic so errors are not the only thing in the log."""
    out = []
    paths = ["/api/products", "/api/cart", "/api/orders", "/health"]
    for m in range(minutes, 0, -1):
        for _ in range(per_min):
            t = ms(m) + rnd.randint(0, 59_000)
            p = rnd.choice(paths)
            out.append(ev(rnd.choice(group_streams), t, app_json(
                "INFO", t, service, f"GET {p} 200 {rnd.randint(4, 60)}ms",
                trace_id=f"{rnd.getrandbits(64):016x}")))
    return out


def metric(namespace, name, dims, fn, stat="Average", period=60, minutes=90, unit="None"):
    """A metric time series. fn(minutes_before) -> value for that minute."""
    stamps, vals = [], []
    for m in range(minutes, -1, -(period // 60)):
        stamps.append(ms(m))
        vals.append(round(fn(m), 2))
    return {"namespace": namespace, "metric": name, "dimensions": dims,
            "stat": stat, "period": period, "unit": unit,
            "timestamps": stamps, "values": vals}


def ramp(start_m, end_m, v0, v1, jitter=0.0):
    """Value that sits at v0, then climbs to v1 between start_m and end_m
    (both measured as 'minutes before the alarm')."""
    def f(m):
        if m >= start_m:
            base = v0
        elif m <= end_m:
            base = v1
        else:
            base = v0 + (v1 - v0) * (start_m - m) / (start_m - end_m)
        return max(0.0, base + rnd.uniform(-jitter, jitter))
    return f


def alarm(name, namespace, metric_name, dims, threshold, op, reason,
          state="ALARM", stat="Average", period=60, eval_periods=3,
          changed_m=0, desc=""):
    """A CloudWatch MetricAlarm (same keys as boto3 describe_alarms)."""
    return {
        "AlarmName": name,
        "AlarmArn": f"arn:aws:cloudwatch:{REGION}:{ACCOUNT}:alarm:{name}",
        "AlarmDescription": desc,
        "StateValue": state,
        "StateReason": reason,
        "StateUpdatedTimestamp": iso(ms(changed_m)),
        "MetricName": metric_name,
        "Namespace": namespace,
        "Statistic": stat,
        "Dimensions": [{"Name": k, "Value": v} for k, v in dims.items()],
        "Period": period,
        "EvaluationPeriods": eval_periods,
        "Threshold": threshold,
        "ComparisonOperator": op,
    }


def ecs_service(cluster, name, taskdef_new, taskdef_old, desired, running,
                pending, failed, deploy_m, events):
    """An ECS service (same keys as boto3 describe_services)."""
    tdarn = lambda rev: f"arn:aws:ecs:{REGION}:{ACCOUNT}:task-definition/{name}:{rev}"
    return {
        "serviceName": name,
        "serviceArn": f"arn:aws:ecs:{REGION}:{ACCOUNT}:service/{cluster}/{name}",
        "clusterArn": f"arn:aws:ecs:{REGION}:{ACCOUNT}:cluster/{cluster}",
        "status": "ACTIVE", "launchType": "FARGATE",
        "desiredCount": desired, "runningCount": running, "pendingCount": pending,
        "taskDefinition": tdarn(taskdef_new),
        "deployments": [
            {"id": f"ecs-svc/{rnd.getrandbits(60)}", "status": "PRIMARY",
             "taskDefinition": tdarn(taskdef_new), "desiredCount": desired,
             "pendingCount": pending, "runningCount": running,
             "failedTasks": failed, "rolloutState": "IN_PROGRESS",
             "createdAt": iso(ms(deploy_m)), "updatedAt": iso(ms(1))},
            {"id": f"ecs-svc/{rnd.getrandbits(60)}", "status": "ACTIVE",
             "taskDefinition": tdarn(taskdef_old), "desiredCount": 0,
             "pendingCount": 0, "runningCount": running, "failedTasks": 0,
             "rolloutState": "COMPLETED",
             "createdAt": iso(ms(60 * 24 * 6)), "updatedAt": iso(ms(deploy_m))},
        ],
        "events": [{"id": f"{rnd.getrandbits(100):x}", "createdAt": iso(ms(m)),
                    "message": msg} for m, msg in events],
    }


# ============ Scenario 1: ECS service dies of OutOfMemory ============
def scenario_ecs_oom():
    svc, group = "checkout-api", "/ecs/checkout-api"
    streams = [f"ecs/{svc}/{rnd.getrandbits(64):016x}" for _ in range(4)]
    logs = noise(streams, svc, 90)
    for m in range(25, 0, -1):                       # after the deploy at -25 min
        sev = (25 - m) / 25                          # problems get worse over time
        for _ in range(1 + int(sev * 8)):
            t = ms(m) + rnd.randint(0, 59_000)
            s = rnd.choice(streams)
            r = rnd.random()
            if r < 0.30 + 0.3 * sev:
                logs.append(ev(s, t, app_json("WARN", t, svc,
                    "GC overhead: heap usage above 92%, full GC took 4.8s",
                    heap_used_mb=rnd.randint(1850, 2040), heap_max_mb=2048)))
            elif r < 0.55 + 0.25 * sev:
                logs.append(ev(s, t, app_json("ERROR", t, svc,
                    "java.lang.OutOfMemoryError: Java heap space",
                    exception="java.lang.OutOfMemoryError",
                    stack="at com.acme.checkout.cache.ProductCache.load(ProductCache.java:142)")))
            else:
                logs.append(ev(s, t, app_json("ERROR", t, svc,
                    f"upstream request failed: 503 Service Unavailable ({rnd.randint(2, 30)}s)",
                    status=503)))
    for m in (18, 14, 9, 5, 2):                      # ECS notices tasks dying
        t = ms(m)
        logs.append(ev(streams[0], t, f"Task stopped: Essential container in task exited, "
                       f"exit code 137 (OOMKilled). Container killed due to memory usage"))
    metrics = [
        metric("AWS/ECS", "MemoryUtilization",
               {"ClusterName": "prod-cluster", "ServiceName": svc},
               ramp(26, 4, 52, 99, 1.5), unit="Percent"),
        metric("AWS/ECS", "CPUUtilization",
               {"ClusterName": "prod-cluster", "ServiceName": svc},
               ramp(26, 4, 35, 78, 3), unit="Percent"),
        metric("AWS/ApplicationELB", "HTTPCode_Target_5XX_Count",
               {"LoadBalancer": "app/prod-alb/50dc6c495c0c9188"},
               ramp(24, 3, 0, 240, 6), stat="Sum", unit="Count"),
    ]
    return {
        "name": "ecs_oom", "anchor_ms": ANCHOR_MS,
        "description": "checkout-api on ECS Fargate runs out of memory after a release that enlarged an in-memory cache.",
        "alarms": [
            alarm("checkout-api-5xx-high", "AWS/ApplicationELB", "HTTPCode_Target_5XX_Count",
                  {"LoadBalancer": "app/prod-alb/50dc6c495c0c9188"}, 50,
                  "GreaterThanThreshold",
                  "Threshold Crossed: 3 datapoints [212.0, 231.0, 240.0] were greater than the threshold (50.0).",
                  stat="Sum", changed_m=2, desc="Checkout 5xx errors from ALB targets"),
            alarm("checkout-api-memory-high", "AWS/ECS", "MemoryUtilization",
                  {"ClusterName": "prod-cluster", "ServiceName": svc}, 85,
                  "GreaterThanThreshold",
                  "Threshold Crossed: 3 datapoints [97.2, 98.9, 99.1] were greater than the threshold (85.0).",
                  changed_m=6, desc="ECS memory above 85%"),
            alarm("checkout-api-cpu-high", "AWS/ECS", "CPUUtilization",
                  {"ClusterName": "prod-cluster", "ServiceName": svc}, 90,
                  "GreaterThanThreshold", "Threshold not crossed.", state="OK", changed_m=300),
        ],
        "log_groups": {group: logs},
        "metrics": metrics,
        "ecs_services": [ecs_service("prod-cluster", svc, 42, 41, 4, 2, 2, 5, 25, [
            (2, f"(service {svc}) has started 2 tasks: (task 9f3a1c) (task 7be201)."),
            (4, f"(service {svc}) (task 5c11de) stopped: Essential container in task exited (exit code 137: OutOfMemoryError)"),
            (9, f"(service {svc}) has stopped 2 running tasks: (task 1ab9f0) (task 66d3e1)."),
            (25, f"(service {svc}) deployment ecs-svc started: task definition {svc}:42"),
        ])],
        "rds_instances": [], "ec2_instances": [],
        "deployments": [
            {"service": svc, "deploymentId": "d-8K2LQ9ZPA", "status": "SUCCEEDED",
             "createdAt": iso(ms(25)), "taskDefinition": f"{svc}:42",
             "image": f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{svc}:v2.14.0",
             "author": "dev-priya", "commit": "a91c3e7",
             "description": "Increase product cache size to hold full catalog in memory (heap 2GB, task memory 2GB)"},
            {"service": svc, "deploymentId": "d-3NQ7H1XKB", "status": "SUCCEEDED",
             "createdAt": iso(ms(60 * 24 * 6)), "taskDefinition": f"{svc}:41",
             "image": f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{svc}:v2.13.2",
             "author": "dev-arjun", "commit": "5be02d4",
             "description": "Fix coupon rounding bug"},
        ],
    }


# ======== Scenario 2: RDS Postgres runs out of connections ===========
def scenario_rds_connections():
    svc, app_group = "orders-service", "/ecs/orders-service"
    db_group = "/aws/rds/instance/orders-db/postgresql"
    app_streams = [f"ecs/{svc}/{rnd.getrandbits(64):016x}" for _ in range(3)]
    db_streams = ["orders-db.postgresql.log.2026-11-18-08"]
    app_logs = noise(app_streams, svc, 90)
    db_logs = []

    def pg(t, user_app, level, msg):
        stamp = datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        ip = f"10.0.{rnd.randint(1, 4)}.{rnd.randint(10, 200)}"
        return f"{stamp} UTC:{ip}({rnd.randint(30000, 60000)}):{user_app}:[{rnd.randint(1000, 9999)}]:{level}:  {msg}"

    for m in range(30, 0, -1):                       # reporting-job starts at -30 min
        sev = (30 - m) / 30
        for _ in range(2 + int(sev * 6)):
            t = ms(m) + rnd.randint(0, 59_000)
            db_logs.append(ev(db_streams[0], t, pg(
                t, "report@orders", "LOG",
                "connection authorized: user=report database=orders application_name=reporting-job")))
        if m <= 12:                                  # the database is now full
            for _ in range(int(sev * 10)):
                t = ms(m) + rnd.randint(0, 59_000)
                db_logs.append(ev(db_streams[0], t, pg(
                    t, "orders_app@orders", "FATAL",
                    "remaining connection slots are reserved for non-replication superuser connections")))
                app_logs.append(ev(rnd.choice(app_streams), t, app_json(
                    "ERROR", t, svc,
                    "HikariPool-1 - Connection is not available, request timed out after 30000ms",
                    exception="java.sql.SQLTransientConnectionException")))
        if m <= 8:
            t = ms(m) + rnd.randint(0, 59_000)
            app_logs.append(ev(rnd.choice(app_streams), t, app_json(
                "ERROR", t, svc, "POST /api/orders 500 Internal Server Error (db unavailable)", status=500)))
    metrics = [
        metric("AWS/RDS", "DatabaseConnections", {"DBInstanceIdentifier": "orders-db"},
               ramp(32, 10, 22, 100, 1), unit="Count"),
        metric("AWS/RDS", "CPUUtilization", {"DBInstanceIdentifier": "orders-db"},
               ramp(32, 10, 28, 41, 3), unit="Percent"),
        metric("AWS/RDS", "FreeableMemory", {"DBInstanceIdentifier": "orders-db"},
               ramp(32, 10, 3.1e9, 1.9e9, 4e7), unit="Bytes"),
    ]
    return {
        "name": "rds_connections", "anchor_ms": ANCHOR_MS,
        "description": "orders-db (RDS PostgreSQL) hits max connections because a new reporting job opens connections without pooling.",
        "alarms": [
            alarm("orders-db-connections-high", "AWS/RDS", "DatabaseConnections",
                  {"DBInstanceIdentifier": "orders-db"}, 90, "GreaterThanThreshold",
                  "Threshold Crossed: 3 datapoints [99.0, 100.0, 100.0] were greater than the threshold (90.0).",
                  changed_m=3, desc="RDS connection count above 90"),
            alarm("orders-service-5xx-high", "AWS/ApplicationELB", "HTTPCode_Target_5XX_Count",
                  {"LoadBalancer": "app/prod-alb/50dc6c495c0c9188"}, 50, "GreaterThanThreshold",
                  "Threshold Crossed: 2 datapoints [88.0, 140.0] were greater than the threshold (50.0).",
                  stat="Sum", changed_m=1),
            alarm("orders-db-cpu-high", "AWS/RDS", "CPUUtilization",
                  {"DBInstanceIdentifier": "orders-db"}, 80, "GreaterThanThreshold",
                  "Threshold not crossed.", state="OK", changed_m=500),
        ],
        "log_groups": {app_group: app_logs, db_group: db_logs},
        "metrics": metrics,
        "ecs_services": [ecs_service("prod-cluster", svc, 77, 76, 3, 3, 0, 0, 60 * 30, [
            (1, f"(service {svc}) has reached a steady state."),
        ])],
        "rds_instances": [{
            "DBInstanceIdentifier": "orders-db", "DBInstanceClass": "db.t3.medium",
            "Engine": "postgres", "EngineVersion": "15.4", "DBInstanceStatus": "available",
            "MultiAZ": True, "AllocatedStorage": 200, "StorageType": "gp3",
            "Endpoint": {"Address": f"orders-db.c9abcdef.{REGION}.rds.amazonaws.com", "Port": 5432},
            "DBParameterGroups": [{"DBParameterGroupName": "orders-pg15", "ParameterApplyStatus": "in-sync"}],
            "PendingModifiedValues": {}, "LatestRestorableTime": iso(ms(5)),
        }],
        "ec2_instances": [],
        "deployments": [
            {"service": "reporting-job", "deploymentId": "d-RJ99AA01", "status": "SUCCEEDED",
             "createdAt": iso(ms(33)), "taskDefinition": "reporting-job:3",
             "image": f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/reporting-job:v1.0.0",
             "author": "dev-meera", "commit": "c0ffee1",
             "description": "New scheduled reporting job; opens a DB connection per report (no pool)"},
        ],
    }


# ================ Scenario 3: EC2 disk is full ======================
def scenario_ec2_disk_full():
    iid = "i-0a1b2c3d4e5f67890"
    app_group, sys_group = "/ec2/app-server/app.log", "/ec2/app-server/syslog"
    stream = iid
    app_logs, sys_logs = [], []
    for m in range(90, 0, -1):
        for _ in range(2):
            t = ms(m) + rnd.randint(0, 59_000)
            app_logs.append(ev(stream, t, app_json("INFO", t, "app-server",
                               f"processed job id={rnd.randint(10000, 99999)} in {rnd.randint(20, 90)}ms")))
        if m <= 70:                                   # DEBUG logging switched on at -70
            for _ in range(3):
                t = ms(m) + rnd.randint(0, 59_000)
                app_logs.append(ev(stream, t, app_json("DEBUG", t, "app-server",
                                   "request payload dump: " + "x" * rnd.randint(200, 400))))
        if m <= 15:
            for _ in range(int((15 - m) / 2) + 1):
                t = ms(m) + rnd.randint(0, 59_000)
                app_logs.append(ev(stream, t, app_json("ERROR", t, "app-server",
                    "OSError: [Errno 28] No space left on device: '/var/log/app/debug.log'",
                    exception="OSError")))
                sys_logs.append(ev(stream, t, f"{datetime.fromtimestamp(t/1000, tz=timezone.utc):%b %d %H:%M:%S} "
                    "ip-10-0-1-55 app[2210]: write failed, filesystem is full"))
    sys_logs.append(ev(stream, ms(70), "Nov 18 07:50:03 ip-10-0-1-55 deploy-agent[998]: applied config change: log_level=DEBUG (ticket OPS-5521)"))
    metrics = [
        metric("CWAgent", "disk_used_percent",
               {"InstanceId": iid, "path": "/", "fstype": "ext4"},
               ramp(70, 8, 48, 100, 0.3), unit="Percent"),
        metric("AWS/EC2", "CPUUtilization", {"InstanceId": iid},
               lambda m: 22 + rnd.uniform(-4, 4), unit="Percent"),
    ]
    return {
        "name": "ec2_disk_full", "anchor_ms": ANCHOR_MS,
        "description": "app-server EC2 disk fills to 100% after DEBUG logging was enabled by a config change.",
        "alarms": [
            alarm("app-server-disk-used-high", "CWAgent", "disk_used_percent",
                  {"InstanceId": iid, "path": "/", "fstype": "ext4"}, 90, "GreaterThanThreshold",
                  "Threshold Crossed: 2 datapoints [99.4, 100.0] were greater than the threshold (90.0).",
                  changed_m=4, desc="Root volume above 90% used"),
            alarm("app-server-cpu-high", "AWS/EC2", "CPUUtilization", {"InstanceId": iid}, 85,
                  "GreaterThanThreshold", "Threshold not crossed.", state="OK", changed_m=900),
        ],
        "log_groups": {app_group: app_logs, sys_group: sys_logs},
        "metrics": metrics,
        "ecs_services": [], "rds_instances": [],
        "ec2_instances": [{
            "InstanceId": iid, "InstanceType": "t3.large", "State": {"Name": "running"},
            "PrivateIpAddress": "10.0.1.55", "LaunchTime": iso(ms(60 * 24 * 40)),
            "Placement": {"AvailabilityZone": f"{REGION}a"},
            "BlockDeviceMappings": [{"DeviceName": "/dev/xvda", "Ebs": {"VolumeId": "vol-0abc123def456", "Status": "attached"}}],
            "Tags": [{"Key": "Name", "Value": "app-server"}, {"Key": "env", "Value": "prod"}],
        }],
        "deployments": [
            {"service": "app-server", "deploymentId": "cfg-OPS-5521", "status": "SUCCEEDED",
             "createdAt": iso(ms(70)), "taskDefinition": "n/a", "image": "n/a",
             "author": "ops-karan", "commit": "n/a",
             "description": "Config change: log_level=DEBUG to investigate a customer issue (no rotation configured for debug.log)"},
        ],
    }


# ============ Scenario 4: bad release (NullPointerException) =========
def scenario_bad_deploy():
    svc, group = "payments-service", "/ecs/payments-service"
    streams = [f"ecs/{svc}/{rnd.getrandbits(64):016x}" for _ in range(3)]
    logs = noise(streams, svc, 90, per_min=3)
    for m in range(12, 0, -1):                        # new release at -12 min
        for _ in range(6 + rnd.randint(0, 4)):
            t = ms(m) + rnd.randint(0, 59_000)
            logs.append(ev(rnd.choice(streams), t, app_json(
                "ERROR", t, svc,
                'java.lang.NullPointerException: Cannot invoke "PaymentRequest.getCurrency()" because "request" is null',
                exception="java.lang.NullPointerException",
                stack="at com.acme.payments.PaymentController.charge(PaymentController.java:87)",
                status=500)))
    metrics = [
        metric("AWS/ApplicationELB", "HTTPCode_Target_5XX_Count",
               {"LoadBalancer": "app/prod-alb/50dc6c495c0c9188"},
               lambda m: (rnd.uniform(0, 3) if m > 12 else rnd.uniform(160, 230)), stat="Sum", unit="Count"),
        metric("AWS/ECS", "CPUUtilization", {"ClusterName": "prod-cluster", "ServiceName": svc},
               lambda m: 30 + rnd.uniform(-3, 3), unit="Percent"),
        metric("AWS/ECS", "MemoryUtilization", {"ClusterName": "prod-cluster", "ServiceName": svc},
               lambda m: 44 + rnd.uniform(-2, 2), unit="Percent"),
    ]
    return {
        "name": "bad_deploy_npe", "anchor_ms": ANCHOR_MS,
        "description": "payments-service release v3.8.0 throws NullPointerException on every charge request.",
        "alarms": [
            alarm("payments-service-5xx-high", "AWS/ApplicationELB", "HTTPCode_Target_5XX_Count",
                  {"LoadBalancer": "app/prod-alb/50dc6c495c0c9188"}, 50, "GreaterThanThreshold",
                  "Threshold Crossed: 3 datapoints [181.0, 205.0, 222.0] were greater than the threshold (50.0).",
                  stat="Sum", changed_m=9, desc="Payments 5xx errors"),
            alarm("payments-service-cpu-high", "AWS/ECS", "CPUUtilization",
                  {"ClusterName": "prod-cluster", "ServiceName": svc}, 90, "GreaterThanThreshold",
                  "Threshold not crossed.", state="OK", changed_m=700),
        ],
        "log_groups": {group: logs},
        "metrics": metrics,
        "ecs_services": [ecs_service("prod-cluster", svc, 118, 117, 3, 3, 0, 0, 12, [
            (11, f"(service {svc}) has reached a steady state."),
            (12, f"(service {svc}) deployment ecs-svc started: task definition {svc}:118"),
        ])],
        "rds_instances": [], "ec2_instances": [],
        "deployments": [
            {"service": svc, "deploymentId": "d-PAY380XQ", "status": "SUCCEEDED",
             "createdAt": iso(ms(12)), "taskDefinition": f"{svc}:118",
             "image": f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{svc}:v3.8.0",
             "author": "dev-rohan", "commit": "7f3d9ab",
             "description": "Refactor PaymentRequest validation and add multi-currency support"},
            {"service": svc, "deploymentId": "d-PAY372MM", "status": "SUCCEEDED",
             "createdAt": iso(ms(60 * 24 * 3)), "taskDefinition": f"{svc}:117",
             "image": f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{svc}:v3.7.2",
             "author": "dev-rohan", "commit": "e41b0c2", "description": "Bump SDK versions"},
        ],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (scenario_ecs_oom, scenario_rds_connections,
               scenario_ec2_disk_full, scenario_bad_deploy):
        data = fn()
        path = OUT_DIR / f"{data['name']}.json"
        path.write_text(json.dumps(data, indent=1))
        n_logs = sum(len(v) for v in data["log_groups"].values())
        print(f"wrote {path.name:24s} {n_logs:5d} log events, "
              f"{len(data['metrics'])} metrics, {len(data['alarms'])} alarms, "
              f"{path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
