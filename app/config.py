import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    SECRET_KEY: str
    ALGORITHM: str
    PROJECT_VOLUME: str
    class Config:
        env_file = ".env"


settings = Settings()


def get_db_url():
    return ("postgresql+asyncpg://admin:password@postgres_fast_api:5432/fast_api")

def get_auth_data():
    return {"secret_key": settings.SECRET_KEY, "algorithm": settings.ALGORITHM}

