-- Casino Intelligence Hub - Real Data Schema
-- Based on actual dataset: all_casino_details2.csv

-- Drop existing tables if they exist
DROP TABLE IF EXISTS raw_casino_data CASCADE;
DROP TABLE IF EXISTS players CASCADE;
DROP TABLE IF EXISTS games CASCADE;
DROP TABLE IF EXISTS daily_player_summary CASCADE;
DROP TABLE IF EXISTS daily_game_summary CASCADE;
DROP TABLE IF EXISTS player_churn_features CASCADE;
DROP TABLE IF EXISTS player_segments CASCADE;
DROP TABLE IF EXISTS anomaly_flags CASCADE;

-- ============================================================================
-- RAW DATA TABLE (matches actual CSV structure)
-- ============================================================================

-- Main casino transactions table (matches your CSV structure exactly)
CREATE TABLE raw_casino_data (
    id BIGSERIAL PRIMARY KEY,
    calc_date DATE NOT NULL,
    user_id INTEGER NOT NULL,
    game_type VARCHAR(50) NOT NULL,
    game_name VARCHAR(100) NOT NULL,
    bet DECIMAL(15,8) NOT NULL,
    won DECIMAL(15,8) NOT NULL,
    profit DECIMAL(15,8) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- DIMENSION TABLES (derived from your real data)
-- ============================================================================

-- Players dimension table
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY,
    first_seen_date DATE NOT NULL,
    last_activity_date DATE NOT NULL,
    total_transactions INTEGER DEFAULT 0,
    total_bet_amount DECIMAL(15,2) DEFAULT 0.00,
    total_won_amount DECIMAL(15,2) DEFAULT 0.00,
    total_profit DECIMAL(15,2) DEFAULT 0.00,
    favorite_game_type VARCHAR(50),
    avg_bet_amount DECIMAL(15,2) DEFAULT 0.00,
    days_active INTEGER DEFAULT 0,
    win_rate DECIMAL(8,4) DEFAULT 0.0000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Games dimension table
CREATE TABLE games (
    game_id SERIAL PRIMARY KEY,
    game_type VARCHAR(50) NOT NULL,
    game_name VARCHAR(100) NOT NULL,
    total_transactions INTEGER DEFAULT 0,
    total_bet_volume DECIMAL(15,2) DEFAULT 0.00,
    total_won_volume DECIMAL(15,2) DEFAULT 0.00,
    house_edge DECIMAL(8,4),
    avg_bet_size DECIMAL(15,2) DEFAULT 0.00,
    unique_players INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(game_type, game_name)
);

-- ============================================================================
-- AGGREGATED FACT TABLES
-- ============================================================================

-- Daily player summary table
CREATE TABLE daily_player_summary (
    player_id INTEGER,
    calc_date DATE,
    transactions_count INTEGER DEFAULT 0,
    total_bet DECIMAL(15,2) DEFAULT 0.00,
    total_won DECIMAL(15,2) DEFAULT 0.00,
    total_profit DECIMAL(15,2) DEFAULT 0.00,
    games_played TEXT[], -- Array of game types played
    avg_bet_size DECIMAL(15,2) DEFAULT 0.00,
    max_bet DECIMAL(15,2) DEFAULT 0.00,
    win_rate DECIMAL(8,4) DEFAULT 0.0000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, calc_date)
);

-- Daily game summary table
CREATE TABLE daily_game_summary (
    game_type VARCHAR(50),
    game_name VARCHAR(100),
    calc_date DATE,
    transactions_count INTEGER DEFAULT 0,
    unique_players INTEGER DEFAULT 0,
    total_bet DECIMAL(15,2) DEFAULT 0.00,
    total_won DECIMAL(15,2) DEFAULT 0.00,
    total_profit DECIMAL(15,2) DEFAULT 0.00,
    avg_bet_size DECIMAL(15,2) DEFAULT 0.00,
    house_edge DECIMAL(8,4) DEFAULT 0.0000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_type, game_name, calc_date)
);

-- ============================================================================
-- MACHINE LEARNING TABLES
-- ============================================================================

-- Player churn features table
CREATE TABLE player_churn_features (
    player_id INTEGER,
    snapshot_date DATE,
    recency_days INTEGER,
    frequency_last_30d INTEGER,
    monetary_last_30d DECIMAL(15,2),
    avg_bet_last_30d DECIMAL(15,2),
    games_diversity_last_30d INTEGER,
    win_rate_last_30d DECIMAL(8,4),
    betting_trend_30d DECIMAL(8,4),
    days_since_last_big_win INTEGER,
    volatility_score DECIMAL(8,4),
    churn_label_next30d BOOLEAN,
    churn_probability DECIMAL(8,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, snapshot_date)
);

-- Player segments table (RFM analysis)
CREATE TABLE player_segments (
    player_id INTEGER,
    segmentation_date DATE,
    rfm_segment VARCHAR(50),
    recency_score INTEGER,
    frequency_score INTEGER,
    monetary_score INTEGER,
    segment_description TEXT,
    customer_lifetime_value DECIMAL(15,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, segmentation_date)
);

-- Anomaly detection results table
CREATE TABLE anomaly_flags (
    anomaly_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    player_id INTEGER NOT NULL,
    detection_date DATE NOT NULL,
    anomaly_type VARCHAR(50) NOT NULL,
    anomaly_score DECIMAL(8,4) NOT NULL,
    is_anomaly BOOLEAN DEFAULT FALSE,
    description TEXT,
    severity VARCHAR(20) DEFAULT 'Medium',
    investigation_status VARCHAR(20) DEFAULT 'Pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE (based on your real data patterns)
-- ============================================================================

-- Primary indexes for raw data
CREATE INDEX idx_raw_casino_data_date ON raw_casino_data(calc_date);
CREATE INDEX idx_raw_casino_data_user ON raw_casino_data(user_id);
CREATE INDEX idx_raw_casino_data_game_type ON raw_casino_data(game_type);
CREATE INDEX idx_raw_casino_data_user_date ON raw_casino_data(user_id, calc_date);
CREATE INDEX idx_raw_casino_data_game_combo ON raw_casino_data(game_type, game_name);

-- Performance indexes for analytics
CREATE INDEX idx_players_last_activity ON players(last_activity_date);
CREATE INDEX idx_players_total_bet ON players(total_bet_amount DESC);
CREATE INDEX idx_players_profit ON players(total_profit DESC);

-- Summary table indexes
CREATE INDEX idx_daily_player_date ON daily_player_summary(calc_date);
CREATE INDEX idx_daily_player_bet ON daily_player_summary(total_bet DESC);
CREATE INDEX idx_daily_game_date ON daily_game_summary(calc_date);
CREATE INDEX idx_daily_game_profit ON daily_game_summary(total_profit DESC);

-- ML table indexes
CREATE INDEX idx_churn_features_date ON player_churn_features(snapshot_date);
CREATE INDEX idx_churn_probability ON player_churn_features(churn_probability DESC);
CREATE INDEX idx_segments_date ON player_segments(segmentation_date);
CREATE INDEX idx_anomaly_flags_date ON anomaly_flags(detection_date);
CREATE INDEX idx_anomaly_flags_player ON anomaly_flags(player_id);

-- ============================================================================
-- VIEWS FOR ANALYTICS (based on your real data)
-- ============================================================================

-- Player performance view
CREATE OR REPLACE VIEW player_analytics AS
SELECT 
    p.player_id,
    p.first_seen_date,
    p.last_activity_date,
    p.total_transactions,
    p.total_bet_amount,
    p.total_won_amount,
    p.total_profit,
    p.avg_bet_amount,
    p.win_rate,
    p.favorite_game_type,
    CURRENT_DATE - p.last_activity_date as days_since_last_activity,
    CASE 
        WHEN CURRENT_DATE - p.last_activity_date <= 7 THEN 'Active'
        WHEN CURRENT_DATE - p.last_activity_date <= 30 THEN 'Recent'
        WHEN CURRENT_DATE - p.last_activity_date <= 90 THEN 'Dormant'
        ELSE 'Churned'
    END as player_status
FROM players p;

-- Game performance view
CREATE OR REPLACE VIEW game_analytics AS
SELECT 
    g.game_type,
    g.game_name,
    g.total_transactions,
    g.total_bet_volume,
    g.total_won_volume,
    g.house_edge,
    g.avg_bet_size,
    g.unique_players,
    g.total_bet_volume - g.total_won_volume as house_profit
FROM games g
ORDER BY house_profit DESC;

-- Success message
SELECT 'Real data schema created successfully!' as status; 