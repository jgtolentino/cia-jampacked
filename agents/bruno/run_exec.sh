#!/bin/bash
# Bruno Secure Executor
# Processes privileged execution tasks from agent_task_queue

# Absolute paths
DB="$HOME/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"
JAMPACKED_HOME="$HOME/Documents/GitHub/jampacked-creative-intelligence"
LIONS_FORGE_HOME="$HOME/Documents/GitHub/pulser-lions-palette-forge"

echo "[Bruno] Starting secure executor..."
echo "[Bruno] Database: $DB"
echo "[Bruno] JamPacked home: $JAMPACKED_HOME"

while true; do
  # Check for pending exec tasks (including approved status)
  TASK=$(sqlite3 "$DB" "SELECT task_id, payload FROM agent_task_queue WHERE target_agent = 'bruno' AND status IN ('pending', 'approved') LIMIT 1;")
  
  if [ -n "$TASK" ]; then
    ID=$(echo "$TASK" | cut -d'|' -f1)
    PAYLOAD=$(echo "$TASK" | cut -d'|' -f2)
    
    # Mark as in progress
    sqlite3 "$DB" "UPDATE agent_task_queue SET status = 'in_progress' WHERE task_id = '$ID';"
    
    echo "[Bruno] Executing task $ID"
    echo "[Bruno] Payload: $PAYLOAD"
    
    # Extract command from payload
    COMMAND=$(echo "$PAYLOAD" | jq -r '.command // empty')
    WORKING_DIR=$(echo "$PAYLOAD" | jq -r '.working_dir // empty')
    
    # Change to working directory if specified
    if [ -n "$WORKING_DIR" ]; then
      echo "[Bruno] Changing to directory: $WORKING_DIR"
      cd "$WORKING_DIR" || {
        echo "[Bruno] Failed to change directory"
        sqlite3 "$DB" "UPDATE agent_task_queue SET status = 'failed', result = '{\"success\": false, \"error\": \"Failed to change directory\"}' WHERE task_id = '$ID';"
        continue
      }
    fi
    
    if [ -n "$COMMAND" ]; then
      # Expanded security whitelist for development tasks
      if [[ "$COMMAND" =~ ^(sudo\ )?(docker|systemctl|service|chmod|npm|node|yarn|pnpm|git|make|./|bash|sh|test|curl|wget) ]] || \
         [[ "$COMMAND" =~ ^(mkdir|cp|mv|rm|touch|echo|cat|ls|pwd|cd) ]] || \
         [[ "$COMMAND" =~ ^(npm\ (install|run|test|build)|yarn\ (install|test|build)|pnpm\ (install|test|build)) ]]; then
        
        echo "[Bruno] Running: $COMMAND"
        
        # Execute with full environment
        export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
        export NODE_ENV="${NODE_ENV:-development}"
        
        # Run command and capture output
        OUTPUT=$(eval "$COMMAND" 2>&1)
        RESULT=$?
        
        echo "[Bruno] Command output: $OUTPUT"
        echo "[Bruno] Exit code: $RESULT"
      else
        echo "[Bruno] Blocked potentially unsafe command: $COMMAND"
        RESULT=1
        OUTPUT="Command blocked by security policy"
      fi
    else
      echo "[Bruno] Invalid payload format - no command found"
      RESULT=1
      OUTPUT="No command specified"
    fi
    
    # Update status with detailed result
    if [ $RESULT -eq 0 ]; then
      # Escape output for JSON
      ESCAPED_OUTPUT=$(echo "$OUTPUT" | jq -Rs .)
      sqlite3 "$DB" "UPDATE agent_task_queue SET status = 'completed', result = '{\"success\": true, \"output\": '"$ESCAPED_OUTPUT"'}' WHERE task_id = '$ID';"
      echo "[Bruno] Task $ID completed successfully"
    else
      ESCAPED_OUTPUT=$(echo "$OUTPUT" | jq -Rs .)
      sqlite3 "$DB" "UPDATE agent_task_queue SET status = 'failed', result = '{\"success\": false, \"error\": '"$ESCAPED_OUTPUT"', \"exit_code\": $RESULT}' WHERE task_id = '$ID';"
      echo "[Bruno] Task $ID failed with exit code: $RESULT"
    fi
  fi
  
  sleep 3
done