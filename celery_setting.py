from celery import Celery

# Celery 애플리케이션 생성
celery_app = Celery(
    "tasks",  # 이름
    broker="redis://localhost:6379/0",  # Redis URL로 브로커 설정
    backend="redis://localhost:6379/0",  # 결과 백엔드 설정
)

# Celery 앱 설정
celery_app.conf.update(
    task_serializer='json',
    result_backend='redis://localhost:6379/0',
)
