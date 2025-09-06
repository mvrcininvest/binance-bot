-- ============================================================================
-- TRADING BOT v9.1 - MANUAL DATABASE MIGRATION
-- ============================================================================
-- This file contains SQL commands to manually upgrade database schema to v9.1
-- Use this if Alembic migration fails or for manual database setup
-- 
-- BACKUP YOUR DATABASE BEFORE RUNNING THESE COMMANDS!
-- ============================================================================

-- ============================================================================
-- 1. TRADES TABLE ENHANCEMENTS
-- ============================================================================

-- Add new columns to trades table
ALTER TABLE trades ADD COLUMN IF NOT EXISTS order_tag VARCHAR(50);
ALTER TABLE trades ADD COLUMN IF NOT EXISTS signal_strength FLOAT DEFAULT 0.5;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS ml_prediction FLOAT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS ml_confidence FLOAT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS position_guard_passed BOOLEAN DEFAULT TRUE;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS leverage_used INTEGER;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS multi_tp_enabled BOOLEAN DEFAULT FALSE;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS tp1_hit BOOLEAN DEFAULT FALSE;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS tp2_hit BOOLEAN DEFAULT FALSE;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS tp3_hit BOOLEAN DEFAULT FALSE;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS trailing_sl_enabled BOOLEAN DEFAULT FALSE;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS max_profit FLOAT DEFAULT 0.0;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS max_drawdown FLOAT DEFAULT 0.0;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS signal_age_seconds INTEGER;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS api_latency_ms INTEGER;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS execution_time_ms INTEGER;

-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_trades_order_tag ON trades(order_tag);
CREATE INDEX IF NOT EXISTS idx_trades_signal_strength ON trades(signal_strength);
CREATE INDEX IF NOT EXISTS idx_trades_leverage_used ON trades(leverage_used);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time_status ON trades(entry_time, status);

-- ============================================================================
-- 2. TRADING SESSIONS TABLE (NEW)
-- ============================================================================

CREATE TABLE IF NOT EXISTS trading_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id VARCHAR(50) UNIQUE NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    mode VARCHAR(20) NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_pnl FLOAT DEFAULT 0.0,
    max_drawdown FLOAT DEFAULT 0.0,
    win_rate FLOAT DEFAULT 0.0,
    profit_factor FLOAT DEFAULT 0.0,
    sharpe_ratio FLOAT,
    max_consecutive_wins INTEGER DEFAULT 0,
    max_consecutive_losses INTEGER DEFAULT 0,
    average_trade_duration_minutes INTEGER,
    total_fees FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON trading_sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_sessions_mode ON trading_sessions(mode);
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON trading_sessions(session_id);

-- ============================================================================
-- 3. PERFORMANCE METRICS TABLE (NEW)
-- ============================================================================

CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_date DATE NOT NULL,
    timeframe VARCHAR(10) NOT NULL, -- 'daily', 'weekly', 'monthly'
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate FLOAT DEFAULT 0.0,
    total_pnl FLOAT DEFAULT 0.0,
    gross_profit FLOAT DEFAULT 0.0,
    gross_loss FLOAT DEFAULT 0.0,
    profit_factor FLOAT DEFAULT 0.0,
    max_drawdown FLOAT DEFAULT 0.0,
    max_drawdown_percent FLOAT DEFAULT 0.0,
    sharpe_ratio FLOAT,
    sortino_ratio FLOAT,
    calmar_ratio FLOAT,
    average_win FLOAT DEFAULT 0.0,
    average_loss FLOAT DEFAULT 0.0,
    largest_win FLOAT DEFAULT 0.0,
    largest_loss FLOAT DEFAULT 0.0,
    max_consecutive_wins INTEGER DEFAULT 0,
    max_consecutive_losses INTEGER DEFAULT 0,
    average_trade_duration_minutes INTEGER,
    total_fees FLOAT DEFAULT 0.0,
    roi_percent FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_metrics_date_timeframe ON performance_metrics(metric_date, timeframe);
CREATE INDEX IF NOT EXISTS idx_metrics_date ON performance_metrics(metric_date);
CREATE INDEX IF NOT EXISTS idx_metrics_timeframe ON performance_metrics(timeframe);

-- ============================================================================
-- 4. ML PREDICTIONS TABLE (NEW)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    prediction_time TIMESTAMP NOT NULL,
    direction VARCHAR(10) NOT NULL, -- 'bullish', 'bearish', 'neutral'
    confidence FLOAT NOT NULL,
    prediction_score FLOAT NOT NULL,
    time_horizon_hours INTEGER NOT NULL,
    patterns_analyzed INTEGER DEFAULT 0,
    trades_used_for_prediction INTEGER DEFAULT 0,
    actual_direction VARCHAR(10), -- filled later for accuracy tracking
    prediction_accuracy FLOAT, -- calculated later
    model_version VARCHAR(20),
    features_used TEXT, -- JSON string of features
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol ON ml_predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_time ON ml_predictions(prediction_time);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_confidence ON ml_predictions(confidence);

-- ============================================================================
-- 5. SYSTEM_LOGS TABLE ENHANCEMENTS
-- ============================================================================

-- Add new columns to system_logs if it exists
ALTER TABLE system_logs ADD COLUMN IF NOT EXISTS log_level VARCHAR(10) DEFAULT 'INFO';
ALTER TABLE system_logs ADD COLUMN IF NOT EXISTS component VARCHAR(50);
ALTER TABLE system_logs ADD COLUMN IF NOT EXISTS session_id VARCHAR(50);
ALTER TABLE system_logs ADD COLUMN IF NOT EXISTS trade_id INTEGER;
ALTER TABLE system_logs ADD COLUMN IF NOT EXISTS execution_time_ms INTEGER;
ALTER TABLE system_logs ADD COLUMN IF NOT EXISTS memory_usage_mb FLOAT;
ALTER TABLE system_logs ADD COLUMN IF NOT EXISTS cpu_usage_percent FLOAT;

-- Add indexes for system_logs
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(log_level);
CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component);
CREATE INDEX IF NOT EXISTS idx_system_logs_session_id ON system_logs(session_id);

-- ============================================================================
-- 6. RISK_MANAGEMENT TABLE (NEW)
-- ============================================================================

CREATE TABLE IF NOT EXISTS risk_management (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    daily_pnl FLOAT DEFAULT 0.0,
    daily_trades INTEGER DEFAULT 0,
    max_position_size_usd FLOAT DEFAULT 0.0,
    total_exposure_usd FLOAT DEFAULT 0.0,
    risk_per_trade_percent FLOAT DEFAULT 2.0,
    max_daily_loss_usd FLOAT DEFAULT 500.0,
    current_drawdown_percent FLOAT DEFAULT 0.0,
    consecutive_losses INTEGER DEFAULT 0,
    risk_level VARCHAR(20) DEFAULT 'NORMAL', -- 'LOW', 'NORMAL', 'HIGH', 'CRITICAL'
    emergency_stop_triggered BOOLEAN DEFAULT FALSE,
    position_limits_hit BOOLEAN DEFAULT FALSE,
    daily_loss_limit_hit BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_risk_management_date ON risk_management(date);
CREATE INDEX IF NOT EXISTS idx_risk_management_risk_level ON risk_management(risk_level);

-- ============================================================================
-- 7. WEBHOOK_LOGS TABLE (NEW)
-- ============================================================================

CREATE TABLE IF NOT EXISTS webhook_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    webhook_id VARCHAR(50) NOT NULL,
    source VARCHAR(50) NOT NULL, -- 'tradingview', 'manual', etc.
    payload TEXT NOT NULL, -- JSON payload
    processed_at TIMESTAMP NOT NULL,
    processing_time_ms INTEGER,
    status VARCHAR(20) NOT NULL, -- 'success', 'error', 'ignored'
    error_message TEXT,
    trade_created BOOLEAN DEFAULT FALSE,
    trade_id INTEGER,
    signal_strength FLOAT,
    position_guard_result VARCHAR(20), -- 'passed', 'failed', 'skipped'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_webhook_logs_webhook_id ON webhook_logs(webhook_id);
CREATE INDEX IF NOT EXISTS idx_webhook_logs_processed_at ON webhook_logs(processed_at);
CREATE INDEX IF NOT EXISTS idx_webhook_logs_status ON webhook_logs(status);
CREATE INDEX IF NOT EXISTS idx_webhook_logs_source ON webhook_logs(source);

-- ============================================================================
-- 8. BOT_STATUS TABLE (NEW)
-- ============================================================================

CREATE TABLE IF NOT EXISTS bot_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status_time TIMESTAMP NOT NULL,
    bot_mode VARCHAR(20) NOT NULL,
    is_trading_enabled BOOLEAN DEFAULT TRUE,
    emergency_stop BOOLEAN DEFAULT FALSE,
    websocket_connected BOOLEAN DEFAULT FALSE,
    api_status VARCHAR(20) DEFAULT 'unknown', -- 'connected', 'disconnected', 'error'
    last_heartbeat TIMESTAMP,
    active_positions INTEGER DEFAULT 0,
    daily_pnl FLOAT DEFAULT 0.0,
    total_balance FLOAT DEFAULT 0.0,
    available_balance FLOAT DEFAULT 0.0,
    memory_usage_mb FLOAT DEFAULT 0.0,
    cpu_usage_percent FLOAT DEFAULT 0.0,
    uptime_seconds INTEGER DEFAULT 0,
    version VARCHAR(10) DEFAULT '9.1',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_bot_status_time ON bot_status(status_time);
CREATE INDEX IF NOT EXISTS idx_bot_status_mode ON bot_status(bot_mode);

-- ============================================================================
-- 9. UPDATE EXISTING DATA (SAFE DEFAULTS)
-- ============================================================================

-- Set default values for new columns in existing trades
UPDATE trades SET 
    signal_strength = 0.5,
    position_guard_passed = TRUE,
    leverage_used = 10,
    multi_tp_enabled = FALSE,
    trailing_sl_enabled = FALSE,
    max_profit = 0.0,
    max_drawdown = 0.0
WHERE signal_strength IS NULL;

-- ============================================================================
-- 10. CREATE VIEWS FOR ANALYTICS
-- ============================================================================

-- Daily performance view
CREATE VIEW IF NOT EXISTS daily_performance AS
SELECT 
    DATE(entry_time) as trade_date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losing_trades,
    ROUND(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
    ROUND(SUM(pnl), 2) as daily_pnl,
    ROUND(AVG(pnl), 2) as avg_pnl_per_trade,
    ROUND(MAX(pnl), 2) as best_trade,
    ROUND(MIN(pnl), 2) as worst_trade
FROM trades 
WHERE status = 'closed' AND is_dry_run = FALSE
GROUP BY DATE(entry_time)
ORDER BY trade_date DESC;

-- Symbol performance view
CREATE VIEW IF NOT EXISTS symbol_performance AS
SELECT 
    symbol,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    ROUND(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
    ROUND(SUM(pnl), 2) as total_pnl,
    ROUND(AVG(pnl), 2) as avg_pnl,
    ROUND(AVG(signal_strength), 3) as avg_signal_strength,
    ROUND(AVG(leverage_used), 1) as avg_leverage
FROM trades 
WHERE status = 'closed' AND is_dry_run = FALSE
GROUP BY symbol
ORDER BY total_pnl DESC;

-- ============================================================================
-- 11. VERIFICATION QUERIES
-- ============================================================================

-- Check if all tables exist
SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;

-- Check trades table structure
PRAGMA table_info(trades);

-- Count records in new tables
SELECT 'trading_sessions' as table_name, COUNT(*) as record_count FROM trading_sessions
UNION ALL
SELECT 'performance_metrics', COUNT(*) FROM performance_metrics
UNION ALL
SELECT 'ml_predictions', COUNT(*) FROM ml_predictions
UNION ALL
SELECT 'risk_management', COUNT(*) FROM risk_management
UNION ALL
SELECT 'webhook_logs', COUNT(*) FROM webhook_logs
UNION ALL
SELECT 'bot_status', COUNT(*) FROM bot_status;

-- ============================================================================
-- END OF MIGRATION
-- ============================================================================

-- Migration completed successfully!
-- Remember to:
-- 1. Backup your database before running this migration
-- 2. Test all functionality after migration
-- 3. Monitor logs for any issues
-- 4. Update your application code to use new fields
-- ============================================================================