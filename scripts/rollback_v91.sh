#!/bin/bash
echo "🔄 Cofanie migracji v9.1..."

# Cofnij migrację bazy danych
docker-compose exec trading-bot alembic downgrade -1

# Przywróć z backupu jeśli potrzeba
if [ -f "backups/pre_v91_backup_*.sql" ]; then
    BACKUP_FILE=$(ls -t backups/pre_v91_backup_*.sql | head -1)
    echo "📦 Przywracanie z backupu: $BACKUP_FILE"
    docker-compose exec -T postgres psql -U bot -d binance_bot < "$BACKUP_FILE"
fi

echo "✅ Cofnięcie zakończone"
