-- ====
-- ALERT_HISTORY v9.1 MIGRATION (PostgreSQL)
-- Dodaje brakujące kolumny używane przez kod webhooka/procesora sygnałów
-- ZRÓB BACKUP zanim uruchomisz!
-- ====

ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS strength FLOAT;
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS indicator_version VARCHAR(20);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS success BOOLEAN;
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS processed_at TIMESTAMP;
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS age_seconds INTEGER;
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS validation_reason TEXT;

-- Indeksy pomocnicze
CREATE INDEX IF NOT EXISTS idx_alert_history_received_at ON alert_history(received_at);
CREATE INDEX IF NOT EXISTS idx_alert_history_processed ON alert_history(processed);
CREATE INDEX IF NOT EXISTS idx_alert_history_version ON alert_history(indicator_version);

-- Weryfikacja
-- SELECT column_name FROM information_schema.columns WHERE table_name='alert_history';