#!/bin/bash

echo "🔍 TRADING BOT v9.1 - WERYFIKACJA PO WDROŻENIU"
echo "=================================================="

# 1. Status usług
echo "📊 1. STATUS USŁUG:"
docker-compose ps
echo ""

# 2. Wykorzystanie zasobów
echo "💾 2. WYKORZYSTANIE ZASOBÓW:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -10
echo ""

# 3. Sprawdzenie logów
echo "📝 3. OSTATNIE LOGI BOTA:"
docker-compose logs --tail=10 trading-bot
echo ""

# 4. Status bazy danych
echo "🗄️ 4. STATUS BAZY DANYCH:"
docker-compose exec -T postgres psql -U trading_user -d trading_bot_v91 -c "SELECT COUNT(*) as total_tables FROM information_schema.tables WHERE table_schema = 'public';"
echo ""

# 5. Łączność z API
echo "🌐 5. ŁĄCZNOŚĆ Z API:"
curl -s http://localhost:5000/health || echo "❌ Health endpoint niedostępny"
echo ""

# 6. Test webhook
echo "🔗 6. TEST WEBHOOK:"
curl -s -X POST http://localhost:5000/webhook \
  -H "Content-Type: application/json" \
  -d '{"action": "TEST", "symbol": "BTCUSDT", "test": true}' || echo "❌ Webhook niedostępny"
echo ""

echo "✅ WERYFIKACJA ZAKOŃCZONA"
