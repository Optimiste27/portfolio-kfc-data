-- Schéma de base de données pour le projet COS KFC
-- Fichier: src/database_schema.sql

-- Table restaurants
CREATE TABLE IF NOT EXISTS restaurants (
    restaurant_id INTEGER PRIMARY KEY,
    restaurant_name VARCHAR(100) NOT NULL,
    region VARCHAR(50),
    opening_date DATE,
    manager_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table product_families
CREATE TABLE IF NOT EXISTS product_families (
    product_family_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_family_name VARCHAR(50) NOT NULL UNIQUE,
    category VARCHAR(50),
    target_gap_percentage DECIMAL(5,2) DEFAULT 4.00
);

-- Table transactions
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    restaurant_id INTEGER NOT NULL,
    product_family_id INTEGER NOT NULL,
    theoretical_unit_cost DECIMAL(10,2) NOT NULL,
    actual_unit_cost DECIMAL(10,2) NOT NULL,
    selling_price DECIMAL(10,2) NOT NULL,
    units_sold INTEGER NOT NULL,
    waste_kg DECIMAL(10,2) DEFAULT 0,
    qsp_score DECIMAL(3,1) CHECK (qsp_score >= 0 AND qsp_score <= 10),
    operational_anomaly BOOLEAN DEFAULT FALSE,
    revenue DECIMAL(15,2),
    theoretical_cos DECIMAL(10,4),
    actual_cos DECIMAL(10,4),
    cos_gap DECIMAL(10,4),
    gap_percentage DECIMAL(10,2),
    gap_severity VARCHAR(10),
    FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id),
    FOREIGN KEY (product_family_id) REFERENCES product_families(product_family_id),
    CHECK (selling_price > 0),
    CHECK (units_sold > 0)
);

-- Table users
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) CHECK (role IN ('viewer', 'manager', 'admin')) DEFAULT 'viewer',
    full_name VARCHAR(100),
    email VARCHAR(100),
    restaurant_id INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id)
);

-- Table kpi_daily
CREATE TABLE IF NOT EXISTS kpi_daily (
    kpi_date DATE NOT NULL,
    restaurant_id INTEGER NOT NULL,
    total_transactions INTEGER DEFAULT 0,
    avg_gap_percentage DECIMAL(10,2),
    total_revenue DECIMAL(15,2),
    total_waste_cost DECIMAL(15,2),
    anomaly_rate DECIMAL(10,2),
    PRIMARY KEY (kpi_date, restaurant_id),
    FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id)
);

-- Table audit_logs
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action_type VARCHAR(50) NOT NULL,
    table_name VARCHAR(50),
    record_id INTEGER,
    old_values TEXT,
    new_values TEXT,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Indexes pour performance
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_transactions_restaurant ON transactions(restaurant_id);
CREATE INDEX IF NOT EXISTS idx_transactions_product ON transactions(product_family_id);
CREATE INDEX IF NOT EXISTS idx_kpi_daily_date ON kpi_daily(kpi_date);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at);

-- Données initiales
INSERT OR IGNORE INTO product_families (product_family_name, category) VALUES
('Poulet', 'Principal'),
('Accompagnements', 'Side'),
('Boissons', 'Beverage'),
('Desserts', 'Dessert'),
('Menus', 'Combo');

-- Utilisateurs par défaut (mot de passe: admin123, manager123, viewer123)
INSERT OR IGNORE INTO users (username, password_hash, role, full_name, email) VALUES
('admin', '240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9', 'admin', 'Administrateur', 'admin@kfc.fr'),
('manager', 'd0e4b4ccbb5c8c6f9c7e8c9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0', 'manager', 'Manager Régional', 'manager@kfc.fr'),
('viewer', '8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92', 'viewer', 'Consultant', 'viewer@kfc.fr');