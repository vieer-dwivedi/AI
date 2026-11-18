# Runbook: Lambda throttling

## Symptoms
- Lambda Throttles metric above zero, function errors with TooManyRequestsException

## Remediation
1. Raise the reserved concurrency for the function. Risk: low.
2. Check account-level concurrency limit and request an increase.
