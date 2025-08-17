import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from database import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def get_url():
    url = os.getenv("DATABASE_URL", "sqlite:////app/data/trading_bot.db")
    return url.strip().strip('"').strip("'")


target_metadata = Base.metadata


def run_migrations_offline():
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        {"sqlalchemy.url": get_url()},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
