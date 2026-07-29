from trade_system.domains.trading.infrastructure.brokers.fyers.client import FyersBroker

def test_fyers_broker_initialization():
    broker = FyersBroker(client_id="cid", access_token="token")
    assert broker.client_id == "cid"
