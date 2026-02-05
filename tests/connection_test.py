#!/usr/bin/env python3
"""
Connection Test Script - Validates API connectivity.
Tests Binance and Polymarket connections before trading.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ccxt
import aiohttp
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


async def test_binance_connection() -> dict:
    """Test Binance API connectivity."""
    result = {
        "name": "Binance",
        "status": "UNKNOWN",
        "latency_ms": None,
        "error": None
    }
    
    try:
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"}
        })
        
        start = datetime.now()
        
        # Test public endpoint
        ticker = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: exchange.fetch_ticker("BTC/USDT")
        )
        
        latency = (datetime.now() - start).total_seconds() * 1000
        
        result["status"] = "OK"
        result["latency_ms"] = round(latency, 2)
        result["price"] = ticker["last"]
        
        log.info(f"✅ Binance: BTC/USDT = ${ticker['last']:,.2f} (latency: {latency:.0f}ms)")
        
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        log.error(f"❌ Binance connection failed: {e}")
        
    return result


async def test_polymarket_connection() -> dict:
    """Test Polymarket CLOB API connectivity."""
    result = {
        "name": "Polymarket",
        "status": "UNKNOWN",
        "latency_ms": None,
        "error": None
    }
    
    try:
        url = "https://clob.polymarket.com/markets"
        
        start = datetime.now()
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params={"limit": 1}) as resp:
                latency = (datetime.now() - start).total_seconds() * 1000
                
                if resp.status == 200:
                    data = await resp.json()
                    result["status"] = "OK"
                    result["latency_ms"] = round(latency, 2)
                    result["market_count"] = len(data) if isinstance(data, list) else "?"
                    log.info(f"✅ Polymarket: API reachable (latency: {latency:.0f}ms)")
                else:
                    result["status"] = "ERROR"
                    result["error"] = f"HTTP {resp.status}"
                    log.error(f"❌ Polymarket: HTTP {resp.status}")
                    
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        log.error(f"❌ Polymarket connection failed: {e}")
        
    return result


async def test_disk_io() -> dict:
    """Test file system read/write permissions."""
    result = {
        "name": "Disk I/O",
        "status": "UNKNOWN",
        "paths_checked": []
    }
    
    paths = [
        Path("data"),
        Path("logs"),
        Path("models"),
    ]
    
    all_ok = True
    
    for path in paths:
        path_status = {"path": str(path), "read": False, "write": False}
        
        try:
            # Create if not exists
            path.mkdir(parents=True, exist_ok=True)
            
            # Test write
            test_file = path / ".connection_test"
            test_file.write_text(f"test {datetime.now()}")
            path_status["write"] = True
            
            # Test read
            _ = test_file.read_text()
            path_status["read"] = True
            
            # Cleanup
            test_file.unlink()
            
            log.info(f"✅ Disk I/O: {path} OK")
            
        except Exception as e:
            all_ok = False
            path_status["error"] = str(e)
            log.error(f"❌ Disk I/O failed for {path}: {e}")
            
        result["paths_checked"].append(path_status)
        
    result["status"] = "OK" if all_ok else "ERROR"
    return result


async def test_secrets() -> dict:
    """Test secrets configuration."""
    result = {
        "name": "Secrets",
        "status": "UNKNOWN",
        "credentials": {}
    }
    
    try:
        from src.infrastructure.secrets import secrets
        
        status = secrets.validate_all()
        result["credentials"] = status
        
        # At least one exchange should be configured
        if status["binance"] or status["polymarket"]:
            result["status"] = "OK"
            log.info(f"✅ Secrets: Binance={status['binance']}, Polymarket={status['polymarket']}")
        else:
            result["status"] = "WARNING"
            log.warning("⚠️ Secrets: No exchange credentials configured")
            
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        log.error(f"❌ Secrets validation failed: {e}")
        
    return result


async def run_all_tests() -> dict:
    """Run all connection tests."""
    log.info("=" * 60)
    log.info("🔌 Starting Connection Tests")
    log.info("=" * 60)
    
    results = {}
    
    # Run tests
    results["binance"] = await test_binance_connection()
    results["polymarket"] = await test_polymarket_connection()
    results["disk_io"] = await test_disk_io()
    results["secrets"] = await test_secrets()
    
    # Summary
    log.info("=" * 60)
    log.info("📊 Test Summary")
    log.info("=" * 60)
    
    all_ok = True
    for name, r in results.items():
        status_icon = "✅" if r["status"] == "OK" else "⚠️" if r["status"] == "WARNING" else "❌"
        log.info(f"  {status_icon} {name}: {r['status']}")
        if r["status"] == "ERROR":
            all_ok = False
            
    log.info("=" * 60)
    
    if all_ok:
        log.info("✅ All critical tests passed. Ready for Phase 2.")
    else:
        log.warning("⚠️ Some tests failed. Review before proceeding.")
        
    return results


if __name__ == "__main__":
    asyncio.run(run_all_tests())
