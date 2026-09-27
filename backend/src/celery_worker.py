import os
from dotenv import load_dotenv
from celery import Celery

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://:redis_password@localhost:6379/0")

celery_app = Celery(
    "triage_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_time_limit=600,  # 10 min hard cap per job
    task_soft_time_limit=540,
    task_reject_on_worker_lost=True,
)

# Import tasks module so Celery discovers them
import tasks  # noqa: E402,F401  (side-effect to register tasks)
