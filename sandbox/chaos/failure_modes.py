"""Failure/fix mappings used by sandbox chaos controller."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FailureSpec:
    """Static metadata for a supported failure mode."""

    service: str
    description: str
    recommended_fix: str


FAILURE_SPECS: dict[str, FailureSpec] = {
    "rate_limiting": FailureSpec(
        service="auth",
        description="Auth throttles incoming requests aggressively.",
        recommended_fix="restart_service",
    ),
    "token_expiry": FailureSpec(
        service="auth",
        description="Token validation fails due to invalid expiry behavior.",
        recommended_fix="config_change",
    ),
    "config_corruption": FailureSpec(
        service="auth",
        description="Service config is invalid and causes unstable behavior.",
        recommended_fix="config_change",
    ),
    "oom": FailureSpec(
        service="database",
        description="Database becomes unavailable due to memory pressure.",
        recommended_fix="memory_increase",
    ),
    "connection_pool_exhaustion": FailureSpec(
        service="database",
        description="Database connection pool is saturated.",
        recommended_fix="restart_service",
    ),
    "replication_lag": FailureSpec(
        service="database",
        description="Reads can become stale or delayed.",
        recommended_fix="config_change",
    ),
    "gateway_timeout": FailureSpec(
        service="payments",
        description="Payments requests timeout before completion.",
        recommended_fix="restart_service",
    ),
    "validation_errors": FailureSpec(
        service="payments",
        description="Payments rejects valid traffic due to bad validation.",
        recommended_fix="config_change",
    ),
    "idempotency_failure": FailureSpec(
        service="payments",
        description="Duplicate transaction safety is disabled.",
        recommended_fix="config_change",
    ),
    "batch_job_runaway": FailureSpec(
        service="analytics",
        description="Analytics worker monopolizes resources.",
        recommended_fix="restart_service",
    ),
    "query_timeout": FailureSpec(
        service="analytics",
        description="Analytics queries fail with timeout behavior.",
        recommended_fix="config_change",
    ),
    "stale_cache": FailureSpec(
        service="analytics",
        description="Analytics serves stale report views.",
        recommended_fix="data_fix",
    ),
    "queue_overflow": FailureSpec(
        service="notifications",
        description="Notification queue depth explodes.",
        recommended_fix="restart_service",
    ),
    "template_error": FailureSpec(
        service="notifications",
        description="Template rendering failures block notifications.",
        recommended_fix="config_change",
    ),
    "rate_exceeded": FailureSpec(
        service="notifications",
        description="Notifications are throttled due to high send rate.",
        recommended_fix="restart_service",
    ),
}


ALLOWED_FIX_TYPES: set[str] = {
    "restart_service",
    "memory_increase",
    "config_change",
    "schema_migration",
    "data_fix",
}

