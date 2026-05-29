import os
import re

MAPPING = {
    r'trade_system\.data': r'trade_system.infrastructure.data',
    r'trade_system\.database': r'trade_system.infrastructure.database',
    r'trade_system\.notifications': r'trade_system.infrastructure.notifications',
    r'trade_system\.brokers': r'trade_system.infrastructure.brokers',
    r'trade_system\.advisory': r'trade_system.application.advisory',
    r'trade_system\.research': r'trade_system.application.research',
    r'trade_system\.backtesting': r'trade_system.application.backtesting',
    r'trade_system\.analysis': r'trade_system.application.analysis',
    r'trade_system\.agent': r'trade_system.application.agent',
    r'trade_system\.indicators': r'trade_system.application.indicators',
    r'trade_system\.pipeline': r'trade_system.application.pipeline',
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
