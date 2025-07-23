# 🔒 JamPacked MCP Persistence Bundle

This bundle ensures Claude Desktop and Claude Code permanently remember your MCP + JamPacked + Drive setup.

## 📦 Bundle Contents

| File | Purpose |
|------|---------|
| `.mcp.yaml` | Master configuration for Claude Code |
| `com.pulser.mcp.sqlite.plist` | Auto-start MCP SQLite server on login |
| `com.pulser.jampacked.services.plist` | Auto-start JamPacked services |
| `start_services.sh` | Service startup script |
| `sync_agent_tasks.sh` | Task queue monitoring |
| `claude-desktop-settings.json` | Claude Desktop configuration |
| `install-persistence.sh` | One-click installer |

## 🚀 Quick Install

```bash
cd /Users/tbwa/Documents/GitHub/jampacked-creative-intelligence/persistence
chmod +x install-persistence.sh
./install-persistence.sh
```

## 🔧 What Gets Configured

### 1. **Auto-Start Services**
- MCP SQLite server starts on login
- JamPacked API, WebSocket, and monitoring services start automatically
- Agent Relay runs continuously

### 2. **Claude Desktop Memory**
- MCP extension stays enabled
- Database path locked in
- SQL shortcuts pre-configured
- Google Drive folder ID saved

### 3. **Claude Code Memory**
- `.mcp.yaml` symlinked to home directory
- Agent paths configured
- Task queue tables defined
- Service endpoints saved

### 4. **Task Monitoring**
- Cron job checks pending tasks every 5 minutes
- Auto-starts agent relay if needed
- Cleans up old completed tasks
- Sends notifications for issues

## 📊 Monitoring

### Check Service Status
```bash
# View all Pulser services
launchctl list | grep pulser

# Check logs
tail -f ~/Library/Logs/jampacked-*.log

# Database status
sqlite3 ~/Documents/GitHub/mcp-sqlite-server/data/database.sqlite "SELECT * FROM task_statistics;"
```

### Task Queue Health
```bash
# Pending tasks
sqlite3 ~/Documents/GitHub/mcp-sqlite-server/data/database.sqlite "SELECT * FROM active_tasks;"

# Agent status
sqlite3 ~/Documents/GitHub/mcp-sqlite-server/data/database.sqlite "SELECT * FROM agent_registry;"
```

## 🛠 Troubleshooting

### Service Won't Start
```bash
# Check error logs
cat ~/Library/Logs/mcp-sqlite-server.error.log
cat ~/Library/Logs/jampacked-services.error.log

# Restart service
launchctl unload ~/Library/LaunchAgents/com.pulser.mcp.sqlite.plist
launchctl load ~/Library/LaunchAgents/com.pulser.mcp.sqlite.plist
```

### Database Connection Issues
```bash
# Verify database exists
ls -la ~/Documents/GitHub/mcp-sqlite-server/data/database.sqlite

# Test connection
sqlite3 ~/Documents/GitHub/mcp-sqlite-server/data/database.sqlite ".tables"
```

### Claude Desktop Not Connecting
1. Open Claude Desktop settings
2. Go to Developer → Extensions
3. Ensure SQLite MCP is enabled
4. Check DB_PATH matches your setup

### Claude Code Not Finding Config
```bash
# Ensure symlink exists
ls -la ~/.mcp.yaml

# Re-create if needed
ln -sf ~/Documents/GitHub/jampacked-creative-intelligence/.mcp.yaml ~/.mcp.yaml
```

## 🔄 Manual Sync

Force a task sync:
```bash
~/Documents/GitHub/jampacked-creative-intelligence/persistence/sync_agent_tasks.sh
```

## 🗑 Uninstall

Remove all persistence:
```bash
# Stop services
launchctl unload ~/Library/LaunchAgents/com.pulser.*.plist

# Remove launch agents
rm ~/Library/LaunchAgents/com.pulser.*.plist

# Remove cron job
crontab -l | grep -v sync_agent_tasks | crontab -

# Remove symlink
rm ~/.mcp.yaml
```

## 🎯 Benefits

✅ **Zero Manual Startup** - Everything runs automatically
✅ **Crash Recovery** - Services restart if they fail
✅ **Task Monitoring** - Never miss a queued task
✅ **Persistent Paths** - No need to reconfigure
✅ **Cross-Session Memory** - Survives restarts

---

Your JamPacked MCP integration is now permanently configured! 🚀