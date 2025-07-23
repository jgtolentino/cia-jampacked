#!/bin/bash
# Deploy Agent Monitoring System

set -e

echo "🚀 Deploying Agent Monitoring System..."
echo "========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 1: Install dependencies
echo "📦 Installing dependencies..."
cd "$SCRIPT_DIR"
npm install

# Step 2: Create LaunchAgent for monitoring server
echo "🔧 Creating LaunchAgent..."
PLIST_PATH="$HOME/Library/LaunchAgents/com.pulser.agent.monitor.plist"

cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pulser.agent.monitor</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/node</string>
        <string>$SCRIPT_DIR/agent-status-server.js</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>HOME</key>
        <string>$HOME</string>
        <key>MONITOR_PORT</key>
        <string>3002</string>
    </dict>
    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$HOME/Library/Logs/agent-monitor.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/Library/Logs/agent-monitor.error.log</string>
</dict>
</plist>
EOF

# Step 3: Load LaunchAgent
echo "⚡ Starting monitoring server..."
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

# Step 4: Create convenience script
echo "📝 Creating convenience commands..."
cat > "$HOME/.agent-monitor" << 'EOF'
# Agent Monitor Commands
alias agent-dash="open http://localhost:3002/dashboard.html"
alias agent-status="curl -s http://localhost:3002/api/mcp/agents/status | jq"
alias agent-tasks="curl -s http://localhost:3002/api/mcp/tasks/recent | jq"
alias agent-stats="curl -s http://localhost:3002/api/mcp/tasks/stats | jq"
alias agent-logs="tail -f ~/Library/Logs/agent-monitor.log"
EOF

# Add to shell profile if not already there
if ! grep -q ".agent-monitor" ~/.zshrc 2>/dev/null; then
    echo "" >> ~/.zshrc
    echo "# Agent Monitor aliases" >> ~/.zshrc
    echo "[ -f ~/.agent-monitor ] && source ~/.agent-monitor" >> ~/.zshrc
fi

# Step 5: Test the deployment
echo ""
echo "🧪 Testing deployment..."
sleep 3

if curl -s http://localhost:3002/health > /dev/null; then
    echo "✅ Monitoring server is running!"
else
    echo "❌ Monitoring server failed to start"
    echo "Check logs: tail -f ~/Library/Logs/agent-monitor.error.log"
    exit 1
fi

# Step 6: Display success message
echo ""
echo "🎉 Agent Monitoring System Deployed!"
echo "========================================="
echo ""
echo "📊 Dashboard: http://localhost:3002/dashboard.html"
echo "🔍 API Endpoints:"
echo "   - http://localhost:3002/api/mcp/agents/status"
echo "   - http://localhost:3002/api/mcp/tasks/stats"
echo "   - http://localhost:3002/api/mcp/tasks/recent"
echo ""
echo "🛠 Useful Commands:"
echo "   agent-dash   - Open dashboard in browser"
echo "   agent-status - Check agent status (JSON)"
echo "   agent-tasks  - View recent tasks"
echo "   agent-stats  - View task statistics"
echo "   agent-logs   - View monitor logs"
echo ""
echo "The dashboard auto-refreshes every 10 seconds!"
echo ""

# Open dashboard
echo "Opening dashboard..."
open http://localhost:3002/dashboard.html