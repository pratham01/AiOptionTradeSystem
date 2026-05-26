# Multi-Broker Architecture Documentation

## Overview

The trading system now supports multiple brokers with automatic failover capabilities. This architecture provides redundancy, flexibility, and improved reliability for market data and trading operations.

## Architecture

### Clean Architecture Pattern

```
src/trade_system/
├── core/                    # Business logic and interfaces (Ports)
│   └── ports/
│       └── broker.py       # Broker interface definition
├── infrastructure/         # External integrations (Adapters)
│   └── brokers/
│       ├── factory.py      # Broker factory and manager
│       ├── fyers/
│       │   └── client.py   # Fyers broker implementation
│       └── dhan/
│           └── client.py   # Dhan broker implementation
└── config/
    └── settings.py         # Configuration management
```

### Key Components

1. **Broker Interface** (`core/ports/broker.py`)
   - Abstract base class defining the contract for all brokers
   - Standardized methods for quotes, historical data, orders
   - Common data structures (MarketQuote, HistoricalData, etc.)

2. **Broker Factory** (`infrastructure/brokers/factory.py`)
   - Creates and manages broker instances
   - Implements primary/backup failover logic
   - Health monitoring and automatic recovery

3. **Broker Implementations**
   - **FyersBroker**: Refactored to implement new interface
   - **DhanBroker**: New implementation for Dhan API

4. **Configuration** (`config/settings.py`)
   - Multi-broker configuration support
   - Environment-based settings
   - Priority-based broker selection

## Usage

### Basic Usage

```python
from trade_system.infrastructure.brokers.factory import get_broker_manager

# Get broker manager (auto-initialized from settings)
manager = get_broker_manager()

# Get quotes with automatic failover
quotes = manager.get_quotes(["NSE:RELIANCE-EQ", "NSE:TCS-EQ"])

# Get historical data
data = manager.get_historical_data(
    symbol="NSE:RELIANCE-EQ",
    start_date=date(2026, 5, 1),
    end_date=date(2026, 5, 6),
    timeframe="DAY"
)
```

### Using Specific Broker

```python
# Use primary broker only
with manager.use_broker("fyers") as broker:
    quotes = broker.get_quotes(["NSE:RELIANCE-EQ"])

# Use backup broker
with manager.use_broker("dhan") as broker:
    data = broker.get_historical_data(...)
```

### Health Monitoring

```python
# Check broker health
health = manager.get_health_status()
for broker_name, status in health.items():
    print(f"{broker_name}: {'Healthy' if status['is_healthy'] else 'Unhealthy'}")

# Authenticate all brokers
results = manager.authenticate_all()
```

## Configuration

### Environment Variables

```bash
# Primary broker selection
PRIMARY_BROKER=fyers          # fyers or dhan
BACKUP_BROKER=dhan            # Optional backup

# Fyers configuration
FYERS_CLIENT_ID=your_client_id
FYERS_USER_ID=your_user_id
FYERS_SECRET_KEY=your_secret
FYERS_ACCESS_TOKEN=your_token
FYERS_ENABLED=true
FYERS_PRIORITY=1

# Dhan configuration
DHAN_CLIENT_ID=your_client_id
DHAN_API_KEY=your_api_key
DHAN_ACCESS_TOKEN=your_token
DHAN_ENABLED=false
DHAN_PRIORITY=2
```

### Settings Class

```python
from trade_system.config.settings import Settings

settings = Settings.load()

# Access broker configurations
fyers_config = settings.fyers
dhan_config = settings.dhan

# Get primary broker config
primary_config = settings.get_primary_broker_config()
```

## Failover Mechanism

### Automatic Failover

1. **Primary Broker First**: Always tries primary broker first
2. **Backup Fallback**: Falls back to backup broker on failure
3. **Any Available**: Tries any healthy broker as last resort
4. **Health Tracking**: Monitors broker health and recovery
5. **Cooldown Period**: Failed brokers enter cooldown before retry

### Health Status

```python
{
    "fyers": {
        "is_healthy": True,
        "last_success": "2026-05-06T07:54:00",
        "last_failure": null,
        "failure_count": 0,
        "authenticated": True
    },
    "dhan": {
        "is_healthy": False,
        "last_success": "2026-05-06T07:50:00",
        "last_failure": "2026-05-06T07:53:00",
        "failure_count": 2,
        "authenticated": False
    }
}
```

## Data Structures

### MarketQuote

```python
@dataclass
class MarketQuote:
    symbol: str
    exchange: str
    last_price: float
    open: float
    high: float
    low: float
    close: float
    previous_close: float
    volume: int
    change: float
    change_percent: float
    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0
    # ... additional fields
```

### HistoricalData

```python
@dataclass
class HistoricalData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
```

## Migration Guide

### From Old System

1. **Update Imports**:
   ```python
   # Old
   from trade_system.brokers.fyers import FyersBroker
   
   # New
   from trade_system.infrastructure.brokers.factory import get_broker_manager
   ```

2. **Replace Broker Instantiation**:
   ```python
   # Old
   broker = FyersBroker(client_id, token)
   
   # New
   manager = get_broker_manager()
   ```

3. **Update Method Calls**:
   ```python
   # Old
   quotes = broker.get_quotes(symbols)
   
   # New
   quotes = manager.get_quotes(symbols)  # With failover
   ```

### Backward Compatibility

The system maintains backward compatibility through:
- Legacy FyersBroker wrapper
- Old method signatures preserved
- Gradual migration path

## Error Handling

### Exception Hierarchy

```
BrokerError
├── AuthenticationError
├── DataFetchError
└── OrderError
```

### Best Practices

```python
try:
    quotes = manager.get_quotes(symbols)
except BrokerError as e:
    logger.error(f"Broker operation failed: {e}")
    # Handle error appropriately
```

## Performance Considerations

### Connection Pooling

- Each broker maintains its own connection
- Automatic reconnection on failures
- Health checks prevent unnecessary retries

### Rate Limiting

- Respects individual broker rate limits
- Implements backoff on failures
- Distributes load across brokers

### Caching

- Consider implementing caching for frequently accessed data
- Cache invalidation on market data updates
- Per-broker cache strategies

## Testing

### Unit Tests

```python
# Test broker factory
def test_broker_factory():
    manager = get_broker_manager()
    assert manager.primary_broker_name == "fyers"

# Test failover
def test_failover():
    # Mock primary broker failure
    # Verify backup is used
```

### Integration Tests

```python
# Test real broker connections
def test_real_brokers():
    manager = get_broker_manager()
    quotes = manager.get_quotes(["NSE:RELIANCE-EQ"])
    assert len(quotes) > 0
```

## Security

### Credential Management

- Store credentials in environment variables
- Use encrypted storage for production
- Rotate access tokens regularly

### API Security

- Validate all API responses
- Implement rate limiting
- Log all broker interactions

## Monitoring

### Metrics to Track

- Broker health status
- API response times
- Failure rates
- Failover frequency

### Logging

```python
import logging
logger = logging.getLogger(__name__)

# Broker operations are logged automatically
# Add custom logging as needed
logger.info(f"Fetching quotes for {symbols}")
```

## Future Enhancements

### Planned Features

1. **Load Balancing**: Distribute requests across multiple brokers
2. **Smart Routing**: Route requests based on data type/quality
3. **Broker Arbitrage**: Compare prices across brokers
4. **WebSocket Support**: Real-time data streaming
5. **More Brokers**: Add support for additional brokers

### Extension Points

- Add new brokers by implementing the Broker interface
- Custom failover strategies
- Plugin architecture for broker-specific features

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Check access tokens
   - Verify API credentials
   - Check token expiration

2. **Import Errors**
   - Ensure PYTHONPATH includes src directory
   - Check relative imports
   - Verify package structure

3. **Failover Not Working**
   - Check backup broker configuration
   - Verify backup broker credentials
   - Check health status

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging for broker operations
```

## Conclusion

The multi-broker architecture provides a robust, flexible foundation for trading operations. It ensures high availability through automatic failover while maintaining clean separation of concerns through the use of design patterns.

For questions or contributions, please refer to the project documentation or create an issue in the repository.
