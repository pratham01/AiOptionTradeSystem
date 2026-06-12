import requests
import csv
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

url = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
res = requests.get(url, headers=headers)

if res.status_code == 200:
    lines = res.text.split('\n')
    symbols = []
    # parse CSV
    reader = csv.reader(lines)
    header_skipped = False
    for row in reader:
        if not header_skipped:
            # wait until we see "UNDERLYING"
            if len(row) > 1 and 'UNDERLYING' in row[0].upper() or 'SYMBOL' in row[1].upper():
                header_skipped = True
            continue
            
        if len(row) > 1:
            sym = row[1].strip()
            if sym and not sym.startswith('Symbol') and sym != 'SYMBOL':
                symbols.append(sym)
                
    # Filter indices if needed (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY)
    indices = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
    stock_symbols = [s for s in symbols if s not in indices]
    print(f"Found {len(stock_symbols)} F&O stocks!")
    print(stock_symbols[:10])
    
    with open('scratch/real_fo_symbols.txt', 'w') as f:
        f.write('\n'.join(stock_symbols))
else:
    print(f"Failed to fetch. Status code: {res.status_code}")
