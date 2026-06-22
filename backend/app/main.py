import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from redis.asyncio import Redis

from app.api import admin, agent, alerts, analytics, approvals, audit, auth, clarification, comments, uploads, websockets
from app.core.config import get_settings
from app.core.security import hash_password
from app.db.session import AsyncSessionLocal, get_db
from app.models import User, UserRole
from app.services.request_security import is_origin_allowed
from app.services.agent_dispatcher import start_dispatcher, stop_dispatcher
from app.services.clarification_expiration import start_cleanup_loop, stop_cleanup_loop
from app.services.canonical_intent import (
    CANONICAL_INTENT_CAPABILITY_VERSION,
    CANONICAL_INTENT_SCHEMA_VERSION,
    INTENT_EXTRACTOR_VERSION,
    INTENT_GROUNDING_VERSION,
    INTENT_NORMALIZER_VERSION,
)
from starlette.formparsers import MultiPartParser
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import time

START_TIME = time.time()

# Raise multipart file-size cap from 1 MB to 50 MB so agent callback
# output files (cleaned .xlsx) are not rejected by the parser.
MultiPartParser.max_file_size = 50 * 1024 * 1024

settings = get_settings()

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

STATE_CHANGING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


@app.middleware("http")
async def enforce_trusted_origin(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api") and request.method.upper() in STATE_CHANGING_METHODS:
        is_service_request = any(
            request.headers.get(header)
            for header in ("x-agent-callback-secret", "x-agent-registry-secret", "x-agent-service-secret")
        )
        origin = (request.headers.get("origin") or "").strip()
        has_cookie_auth = bool(request.cookies.get("access_token") or request.cookies.get("refresh_token"))
        if origin:
            if not is_origin_allowed(origin):
                return JSONResponse(status_code=403, content={"detail": "Origin is not allowed"})
        elif has_cookie_auth and not is_service_request:
            return JSONResponse(status_code=403, content={"detail": "Missing Origin header"})
    return await call_next(request)

app.include_router(uploads.router, prefix="/api")
app.include_router(clarification.router, prefix="/api")
app.include_router(agent.router, prefix="/api")
app.include_router(approvals.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(admin.router, prefix="/api")
app.include_router(comments.router, prefix="/api")
app.include_router(audit.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(websockets.router)


@app.on_event("startup")
async def on_startup() -> None:
    if settings.environment.lower() == "development" or settings.seed_default_users:
        async with AsyncSessionLocal() as db:
            await seed_user(
                db,
                name=settings.default_admin_name,
                email=settings.default_admin_email,
                password=settings.default_admin_password,
                role=UserRole.admin,
            )
            if settings.agent_email and settings.agent_password:
                await seed_user(
                    db,
                    name=settings.agent_name,
                    email=settings.agent_email,
                    password=settings.agent_password,
                    role=UserRole.employee,
                )
            if settings.environment.lower() == "development":
                await ensure_demo_reporting_lines(db)
    await start_dispatcher(app)
    await start_cleanup_loop(app)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await stop_dispatcher(app)
    await stop_cleanup_loop(app)


async def seed_user(db, *, name: str, email: str, password: str, role: UserRole) -> None:
    existing_user = (
        await db.execute(select(User).where(User.email == email.lower()))
    ).scalar_one_or_none()
    if existing_user is not None:
        return

    db.add(
        User(
            full_name=name,
            email=email.lower(),
            hashed_password=hash_password(password),
            role=role,
        )
    )
    await db.commit()


async def ensure_demo_reporting_lines(db) -> None:
    manager = (
        await db.execute(select(User).where(User.email == "kukretimanas8@gmail.com"))
    ).scalar_one_or_none()
    if manager is None:
        return

    employees = (
        await db.execute(
            select(User).where(User.email.in_(["employee@gmail.com", "employee1@gmail.com"]))
        )
    ).scalars().all()

    updated = False
    for employee in employees:
        if employee.manager_id == manager.id:
            continue
        employee.manager_id = manager.id
        updated = True

    if updated:
        await db.commit()


@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)) -> dict[str, str | int | bool | float]:
    payload: dict[str, str | int | bool | float] = {
        "status": "ok",
        "version": app.version,
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "environment": getattr(settings, "environment", "development"),
        "canonical_intent_schema_version": CANONICAL_INTENT_SCHEMA_VERSION,
        "capability_version": CANONICAL_INTENT_CAPABILITY_VERSION,
        "intent_extractor_version": INTENT_EXTRACTOR_VERSION,
        "intent_normalizer_version": INTENT_NORMALIZER_VERSION,
        "intent_grounding_version": INTENT_GROUNDING_VERSION,
        "compiler_version": "1.0",
        "standalone_agent_planning_enabled": False,
    }

    try:
        await db.execute(text("SELECT 1"))
        payload["database"] = "ok"
    except Exception:
        payload["database"] = "unreachable"
        payload["status"] = "degraded"

    redis = getattr(app.state, "agent_dispatch_redis", None)
    if redis is None:
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        close_redis = True
    else:
        close_redis = False

    try:
        await redis.ping()
        payload["redis"] = "ok"
        payload["dispatch_queue_depth"] = int(await redis.llen(settings.agent_dispatch_queue))
        payload["dispatch_retry_depth"] = int(await redis.zcard(settings.agent_dispatch_retry_queue))
        payload["dispatch_dead_letter_depth"] = int(await redis.llen(settings.agent_dead_letter_queue))
        payload["dispatcher_running"] = bool(getattr(app.state, "agent_dispatch_task", None))
    except Exception:
        payload["redis"] = "unreachable"
        payload["status"] = "degraded"
    finally:
        if close_redis:
            await redis.aclose()

    return payload
