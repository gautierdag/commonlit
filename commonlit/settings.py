from pydantic import BaseSettings


class Settings(BaseSettings):
    wandb_api_key: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


env_settings = Settings()
