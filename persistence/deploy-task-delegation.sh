#!/bin/bash
# Deploy Task Delegation System
# Enables Claude Desktop to delegate tasks to Bruno and Pulser CLI

set -e

echo "🚀 Deploying Task Delegation System..."
echo "========================================="

# Configuration
DB_PATH="$HOME/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Step 1: Create task queue schema
echo "📊 Creating task queue schema..."
if [ -f "$DB_PATH" ]; then
  sqlite3 "$DB_PATH" < "$SCRIPT_DIR/create-task-queue-schema.sql"
  echo "✅ Task queue schema created"
else
  echo "❌ Database not found at: $DB_PATH"
  echo "Please ensure MCP SQLite server is installed"
  exit 1
fi

# Step 2: Create Bruno executor directory
echo "🔨 Setting up Bruno executor..."
BRUNO_DIR="$PROJECT_ROOT/agents/bruno"
mkdir -p "$BRUNO_DIR"

# Create Bruno execution script
cat > "$BRUNO_DIR/run_exec.sh" << 'EOF'
#!/bin/bash
# Bruno Secure Executor
# Processes privileged execution tasks from agent_task_queue

DB="$HOME/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"

echo "[Bruno] Starting secure executor..."

while true; do
  # Check for pending exec tasks
  TASK=$(sqlite3 "$DB" "SELECT task_id, payload FROM agent_task_queue WHERE target_agent = 'bruno' AND status = 'pending' LIMIT 1;")
  
  if [ -n "$TASK" ]; then
    ID=$(echo "$TASK" | cut -d'|' -f1)
    PAYLOAD=$(echo "$TASK" | cut -d'|' -f2)
    
    # Mark as in progress
    sqlite3 "$DB" "UPDATE agent_task_queue SET status = 'in_progress' WHERE task_id = '$ID';"
    
    echo "[Bruno] Executing task $ID"
    
    # Extract command from payload
    COMMAND=$(echo "$PAYLOAD" | jq -r '.command // empty')
    
    if [ -n "$COMMAND" ]; then
      # Security check: Only allow specific sudo commands
      if [[ "$COMMAND" =~ ^sudo\ (docker|systemctl|service) ]]; then
        echo "[Bruno] Running: $COMMAND"
        eval "$COMMAND"
        RESULT=$?
      else
        echo "[Bruno] Blocked unsafe command: $COMMAND"
        RESULT=1
      fi
    else
      echo "[Bruno] Invalid payload format"
      RESULT=1
    fi
    
    # Update status
    if [ $RESULT -eq 0 ]; then
      sqlite3 "$DB" "UPDATE agent_task_queue SET status = 'completed', result = '{\"success\": true}' WHERE task_id = '$ID';"
    else
      sqlite3 "$DB" "UPDATE agent_task_queue SET status = 'failed', result = '{\"success\": false, \"error\": \"Command failed\"}' WHERE task_id = '$ID';"
    fi
  fi
  
  sleep 3
done
EOF

chmod +x "$BRUNO_DIR/run_exec.sh"
echo "✅ Bruno executor created"

# Step 3: Create LaunchAgent for Pulser task runner
echo "🎯 Creating Pulser CLI LaunchAgent..."
PLIST_PATH="$HOME/Library/LaunchAgents/com.pulser.pulser.taskrunner.plist"

cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pulser.pulser.taskrunner</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$SCRIPT_DIR/pulser-task-runner.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_ROOT</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$HOME/Library/Logs/pulser-taskrunner.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/Library/Logs/pulser-taskrunner.error.log</string>
</dict>
</plist>
EOF

# Load the LaunchAgent
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"
echo "✅ Pulser task runner LaunchAgent installed"

# Step 4: Update symlink for Claude Code
echo "🔗 Updating Claude Code configuration..."
ln -sf "$PROJECT_ROOT/.mcp.yaml" "$HOME/.mcp.yaml"
echo "✅ Claude Code configuration updated"

# Step 5: Test the setup
echo ""
echo "🧪 Testing task delegation..."
echo "========================================="

# Insert test tasks
TEST_ID="test_$(date +%s)"
sqlite3 "$DB_PATH" << EOF
INSERT INTO agent_task_queue
(task_id, source_agent, target_agent, task_type, payload)
VALUES
('${TEST_ID}_bruno', 'claude_desktop', 'bruno', 'exec', '{"command": "echo Bruno test successful"}'),
('${TEST_ID}_pulser', 'claude_desktop', 'pulser_cli', 'orchestrate', '{"plan": "test orchestration"}');
EOF

echo "✅ Test tasks inserted"
echo ""

# Display status
echo "📊 Task Queue Status:"
sqlite3 "$DB_PATH" -header -column "SELECT task_id, target_agent, status FROM agent_task_queue WHERE task_id LIKE '${TEST_ID}%';"

echo ""
echo "🎉 Task Delegation System Deployed!"
echo "========================================="
echo ""
echo "✅ Bruno agent configured for secure execution"
echo "✅ Pulser CLI task runner active"
echo "✅ Claude Desktop can now delegate tasks"
echo "✅ Task queue schema created"
echo ""
echo "📝 Example Usage from Claude Desktop:"
echo ""
echo "-- Delegate Docker operation to Bruno:"
echo "INSERT INTO agent_task_queue"
echo "(task_id, source_agent, target_agent, task_type, payload)"
echo "VALUES ('docker_001', 'claude_desktop', 'bruno', 'exec',"
echo "        '{\"command\": \"sudo docker ps\"}');"
echo ""
echo "-- Delegate orchestration to Pulser:"
echo "INSERT INTO agent_task_queue"
echo "(task_id, source_agent, target_agent, task_type, payload)"
echo "VALUES ('orch_001', 'claude_desktop', 'pulser_cli', 'orchestrate',"
echo "        '{\"plan\": \"run scout data pipeline\"}');"
echo ""
echo "🔍 Monitor logs:"
echo "tail -f ~/Library/Logs/pulser-taskrunner.log"
echo ""