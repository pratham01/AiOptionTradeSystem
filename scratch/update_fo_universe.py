import requests
import csv
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

res = requests.get("https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv", headers=headers)

if res.status_code == 200:
    lines = res.text.split('\n')
    symbols = []
    reader = csv.reader(lines)
    header_skipped = False
    for row in reader:
        if not header_skipped:
            if len(row) > 1 and ('UNDERLYING' in row[0].upper() or 'SYMBOL' in row[1].upper()):
                header_skipped = True
            continue
            
        if len(row) > 1:
            sym = row[1].strip()
            if sym and not sym.startswith('Symbol') and sym != 'SYMBOL':
                symbols.append(sym)
                
    indices = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50']
    stock_symbols = [s for s in symbols if s not in indices]
    print(f"Found {len(stock_symbols)} real F&O stocks!")
    
    # Load existing json
    with open('config/fo_universe.json', 'r') as f:
        old_data = json.load(f)
        
    new_data = {}
    added = 0
    removed = 0
    
    for s in stock_symbols:
        fyers_sym = f"NSE:{s}-EQ"
        if fyers_sym in old_data:
            new_data[fyers_sym] = old_data[fyers_sym]
        else:
            new_data[fyers_sym] = "UNKNOWN"
            added += 1
            
    for old_s in old_data.keys():
        if old_s not in new_data:
            removed += 1
            
    with open('config/fo_universe.json', 'w') as f:
        json.dump(new_data, f, indent=4)
        
    print(f"Updated fo_universe.json! Added {added}, Removed {removed}.")
else:
    print(f"Failed to fetch CSV. Status code: {res.status_code}")
