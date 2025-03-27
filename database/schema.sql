-- Users table for authentication
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    two_factor_secret VARCHAR(32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Polling Companies
CREATE TABLE polling_companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    active BOOLEAN DEFAULT true
);

-- Historical Polling Data
CREATE TABLE historical_polls (
    id SERIAL PRIMARY KEY,
    polling_company_id INTEGER REFERENCES polling_companies(id),
    riding_number INTEGER NOT NULL,
    riding_name VARCHAR(100) NOT NULL,
    poll_date DATE NOT NULL,
    party_name VARCHAR(50) NOT NULL,
    support_percentage DECIMAL(5,2) NOT NULL,
    sample_size INTEGER,
    margin_of_error DECIMAL(4,2)
);

-- Star Candidates
CREATE TABLE star_candidates (
    id SERIAL PRIMARY KEY,
    riding_number INTEGER NOT NULL,
    election_year INTEGER NOT NULL,
    candidate_name VARCHAR(100) NOT NULL,
    party_name VARCHAR(50) NOT NULL,
    performance_delta DECIMAL(5,2) NOT NULL,  -- How much they over/underperformed their party
    previous_role VARCHAR(100),
    won BOOLEAN
);

-- Prediction Results
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    riding_number INTEGER NOT NULL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    party_name VARCHAR(50) NOT NULL,
    predicted_percentage DECIMAL(5,2) NOT NULL,
    confidence_interval DECIMAL(4,2),
    star_candidate_adjusted BOOLEAN DEFAULT false
);