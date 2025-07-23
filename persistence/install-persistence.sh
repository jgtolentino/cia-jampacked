#!/bin/bash

# JamPacked MCP Persistence Installer
# Sets up automatic startup and persistent configuration

echo "🚀 Installing JamPacked MCP Persistence Bundle"
echo "============================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Paths
PERSISTENCE_DIR="$(dirname "$0")"
LAUNCHAGENTS_DIR="$HOME/Library/LaunchAgents"
JAMPACKED_HOME="$(dirname "$PERSISTENCE_DIR")"

# Function to install LaunchAgent
install_launch_agent() {
    local plist_name=$1
    local source_file="$PERSISTENCE_DIR/$plist_name"
    local dest_file="$LAUNCHAGENTS_DIR/$plist_name"
    
    echo "Installing $plist_name..."
    
    # Update paths in plist if needed
    if [[ "$plist_name" == "com.pulser.mcp.sqlite.plist" ]]; then
        # Check for node path
        NODE_PATH=$(which node)
        if [ -z "$NODE_PATH" ]; then
            echo -e "${YELLOW}⚠️  Node.js not found in PATH${NC}"
            echo "   Please install Node.js first"
            return 1
        fi
        
        # Update node path in plist
        sed "s|/usr/local/bin/node|$NODE_PATH|g" "$source_file" > "$dest_file"
    else
        cp "$source_file" "$dest_file"
    fi
    
    # Load the agent
    launchctl unload "$dest_file" 2>/dev/null
    launchctl load "$dest_file"
    
    echo -e "${GREEN}✅ Installed and loaded $plist_name${NC}"
}

# 1. Make scripts executable
echo "Setting permissions..."
chmod +x "$PERSISTENCE_DIR/start_services.sh"
chmod +x "$PERSISTENCE_DIR/sync_agent_tasks.sh"

# 2. Create logs directory
echo "Creating log directories..."
mkdir -p "$HOME/Library/Logs"

# 3. Install LaunchAgents
echo ""
echo "Installing LaunchAgents..."
mkdir -p "$LAUNCHAGENTS_DIR"

install_launch_agent "com.pulser.mcp.sqlite.plist"
install_launch_agent "com.pulser.jampacked.services.plist"

# 4. Set up cron job for task sync
echo ""
echo "Setting up task sync cron job..."
CRON_CMD="*/5 * * * * $PERSISTENCE_DIR/sync_agent_tasks.sh"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "sync_agent_tasks.sh"; then
    echo -e "${YELLOW}⚠️  Cron job already exists${NC}"
else
    # Add cron job
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo -e "${GREEN}✅ Added cron job for task sync (every 5 minutes)${NC}"
fi

# 5. Create symlink to .mcp.yaml in home directory (for Claude Code)
echo ""
echo "Creating configuration symlinks..."
if [ ! -f "$HOME/.mcp.yaml" ]; then
    ln -s "$JAMPACKED_HOME/.mcp.yaml" "$HOME/.mcp.yaml"
    echo -e "${GREEN}✅ Created ~/.mcp.yaml symlink${NC}"
else
    echo -e "${YELLOW}⚠️  ~/.mcp.yaml already exists${NC}"
fi

# 6. Display status
echo ""
echo "📊 Installation Status:"
echo "====================="

# Check if services are running
if launchctl list | grep -q "com.pulser.mcp.sqlite"; then
    echo -e "${GREEN}✅ MCP SQLite Server: Running${NC}"
else
    echo -e "${YELLOW}⚠️  MCP SQLite Server: Not running${NC}"
fi

if launchctl list | grep -q "com.pulser.jampacked.services"; then
    echo -e "${GREEN}✅ JamPacked Services: Running${NC}"
else
    echo -e "${YELLOW}⚠️  JamPacked Services: Not running${NC}"
fi

# 7. Test database connection
echo ""
echo "Testing database connection..."
if sqlite3 "$HOME/Documents/GitHub/mcp-sqlite-server/data/database.sqlite" "SELECT COUNT(*) FROM agent_registry;" >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Database connection: OK${NC}"
else
    echo -e "${YELLOW}⚠️  Database connection: Failed${NC}"
fi

# 8. Instructions
echo ""
echo "🎯 Next Steps:"
echo "============="
echo "1. Claude Desktop: Extension should auto-connect to MCP"
echo "2. Claude Code: Use 'claude code --config ~/.mcp.yaml'"
echo "3. Monitor logs: tail -f ~/Library/Logs/jampacked-*.log"
echo "4. Check status: launchctl list | grep pulser"
echo ""
echo "✅ Persistence setup complete!"
echo ""
echo "To uninstall:"
echo "  launchctl unload ~/Library/LaunchAgents/com.pulser.*.plist"
echo "  rm ~/Library/LaunchAgents/com.pulser.*.plist"
echo "  crontab -l | grep -v sync_agent_tasks | crontab -"