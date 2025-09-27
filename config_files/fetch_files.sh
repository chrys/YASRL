REMOTE_USER="chrys"
REMOTE_HOST="myVPS3"
FILES=(
	    "/etc/nginx/sites-enabled/fasolaki.com"
		"/etc/systemd/system/yasrl-api.service"
	)
	
# === SCP Operation Loop ===
for FILE in "${FILES[@]}"; do
	echo "Transferring $FILE"
	scp "$REMOTE_USER@$REMOTE_HOST:$FILE" "."
	if [ $? -eq 0 ]; then
	    echo "$FILE transferred successfully."
	else
	    echo "Failed to transfer $FILE."
	fi
done
