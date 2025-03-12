# celery_settings.py

from celery import Celery
import multiprocessing

celery_app = Celery(
    'tasks',  # Celery 애플리케이션 이름
    broker='redis://localhost:6379/0',  # Redis URL로 브로커 설정
    backend='redis://localhost:6379/0',  # 결과 백엔드 설정
)

celery_app.conf.update(
    task_serializer='json',
    result_backend='redis://localhost:6379/0',
    worker_pool='solo',  # solo 대신 spawn 사용
)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    celery_app.start()
