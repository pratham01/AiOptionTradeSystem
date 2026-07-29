import os
import re

MAPPING = {
    r'trade_system\.data': r'trade_system.domains.market_data.infrastructure.data',
    r'trade_system\.database': r'trade_system.domains.market_data.infrastructure.database',
    r'trade_system\.notifications': r'trade_system.shared.notifications',
    r'trade_system\.brokers': r'trade_system.domains.trading.infrastructure.brokers',
    r'trade_system\.advisory': r'trade_system.domains.advisory.application.advisory',
    r'trade_system\.research': r'trade_system.domains.analysis.application.research',
    r'trade_system\.backtesting': r'trade_system.domains.analysis.application.backtesting',
    r'trade_system\.analysis': r'trade_system.domains.analysis.application.analysis',
    r'trade_system\.agent': r'trade_system.domains.advisory.application.agent',
    r'trade_system\.indicators': r'trade_system.domains.strategy.application.indicators',
    r'trade_system\.pipeline': r'trade_system.domains.trading.application.pipeline',
    r'trade_system\.live': r'trade_system.interfaces.live',
    r'trade_system\.dashboard': r'trade_system.interfaces.dashboard',
}

def fix_imports(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith('.py'):
                continue
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            new_content = content
            for old, new in MAPPING.items():
                new_content = re.sub(old, new, new_content)
                
            if new_content != content:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Fixed imports in {path}")

if __name__ == "__main__":
    fix_imports('tests/')
