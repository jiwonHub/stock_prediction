SELECT COUNT(*) AS stock_count
FROM stocks;

SELECT
    code,
    name,
    market,
    current_price,
    change_rate,
    market_cap,
    last_price_at
FROM stocks
ORDER BY market_cap DESC NULLS LAST
LIMIT 20;

SELECT
    stock_code,
    trade_date,
    open,
    high,
    low,
    close,
    volume
FROM stock_prices
WHERE stock_code = '005930'
ORDER BY trade_date DESC
LIMIT 20;

SELECT
    stock_code,
    business_year,
    report_code,
    fs_div,
    sj_div,
    account_nm,
    thstrm_amount
FROM financial_statements
WHERE stock_code = '005930'
ORDER BY id
LIMIT 100;
