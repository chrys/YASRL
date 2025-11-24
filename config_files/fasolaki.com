# HTTP Server - Redirect everything to HTTPS

server {
    if ($host = www.fasolaki.com) {
        return 301 https://$host$request_uri;
    } # managed by Certbot


    if ($host = fasolaki.com) {
        return 301 https://$host$request_uri;
    } # managed by Certbot



        listen 80 default_server;

        server_name fasolaki.com www.fasolaki.com;
	client_max_body_size 20M; 

        return 301 https://$host$request_uri;





}


# HTTPS Server

server {

	listen 443 ssl;
	server_name fasolaki.com www.fasolaki.com;
	client_max_body_size 20M;
	# SSL Configuration
    	ssl_certificate /etc/letsencrypt/live/fasolaki.com/fullchain.pem; # managed by Certbot
    	ssl_certificate_key /etc/letsencrypt/live/fasolaki.com/privkey.pem; # managed by Certbot
        include /etc/letsencrypt/options-ssl-nginx.conf;

        # Django Chatbot App

        location /chatbot/ {
    	rewrite /chatbot/(.*) /$1 break;
    	proxy_pass http://127.0.0.1:7860/;
    	proxy_set_header Host $host;
    	proxy_set_header X-Real-IP $remote_addr;
    	proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    	proxy_set_header X-Forwarded-Proto $scheme;
    	proxy_set_header Upgrade $http_upgrade;
    	proxy_set_header Connection "upgrade";
    	proxy_read_timeout 300s;
    	proxy_send_timeout 300s;
	}

	location /marlensplace/ {
		alias /srv/marlensplace/;
		index index.html;
		try_files $uri $uri/ /marlensplace/index.html;
	}	

	location /my_chatbot/ {
                proxy_pass http://unix:/run/fasolaki-chatbot/fasolaki-chatbot.sock;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
                proxy_set_header Upgrade $http_upgrade;
                proxy_set_header Connection "upgrade";
        }

	# YASRL API (FastAPI) - MUST come before /my_chatbot2/ for correct routing
	location /my_chatbot2/api/ {
    		proxy_pass http://unix:/run/yasrl-api/yasrl-api.sock;
    		proxy_set_header Host $host;
    		proxy_set_header X-Real-IP $remote_addr;
    		proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    		proxy_set_header X-Forwarded-Proto $scheme;
    		proxy_set_header X-Forwarded-Prefix /my_chatbot2/api;
    		proxy_set_header Upgrade $http_upgrade;
    		proxy_set_header Connection "upgrade";
    
    		proxy_connect_timeout 300s;
    		proxy_send_timeout 300s;
    		proxy_read_timeout 300s;
    
    		proxy_buffering on;
    		proxy_buffer_size 128k;
    		proxy_buffers 4 256k;
    		proxy_busy_buffers_size 256k;
	}

	# YASRL UI (Flask) - WITH REWRITE to strip prefix
	location /my_chatbot2/ {
    		rewrite ^/my_chatbot2(/.*)?$ $1 break;
    		proxy_pass http://unix:/run/yasrl-ui/yasrl-ui.sock;
   		proxy_set_header Host $host;
    		proxy_set_header X-Real-IP $remote_addr;
    		proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    		proxy_set_header X-Forwarded-Proto $scheme;
    		proxy_set_header X-Forwarded-Prefix /my_chatbot2;
    		proxy_set_header Upgrade $http_upgrade;
    		proxy_set_header Connection "upgrade";
    
    		# Timeouts for long-running operations
    		proxy_connect_timeout 300s;
    		proxy_send_timeout 300s;
    		proxy_read_timeout 300s;
    
    		# Buffer settings
    		proxy_buffering on;
    		proxy_buffer_size 128k;
    		proxy_buffers 4 256k;
    		proxy_busy_buffers_size 256k;
	}


	# Static files for chatbot
    	location /my_chatbot/static/ {
        	alias /srv/my_apps/static/;
        	expires 30d;
        	access_log off;
        	add_header Cache-Control "public, immutable";
    	}

    	# Media files for chatbot
    	location /my_chatbot/media/ {
        	alias /srv/my_apps/media/;
        	expires 7d;
        	access_log off;
    	}

        # Django App

        location / {
                proxy_pass http://unix:/run/fasolaki.sock;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
        }


        # Serve static files for Django

        location /static/ {

           alias /srv/My-Personal-Website/static/;

        }


        # Serve media files for Django

        location /media/ {

           alias /srv/My-Personal-Website/media/;

        }


        # WordPress configuration
	location ^~ /wordpress/ {
	    root /srv;
	    index index.php index.html index.htm;
	    try_files $uri $uri/ /wordpress/index.php?$args;

	    location ~ \.php$ {
	        fastcgi_split_path_info ^(.+\.php)(/.+)$;
	        fastcgi_pass unix:/var/run/php/php8.3-fpm.sock;
	        fastcgi_index index.php;
	        include fastcgi_params;
	        fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
	        fastcgi_param PATH_INFO $fastcgi_path_info;
	    }
	}


        # YourPlanner static files

        location /yourplanner/static/ {

                alias /srv/yourplanner/staticfiles_production/;
                expires 30d;
                access_log off;

        }


        # YourPlanner media files

        location /yourplanner/media/ {

                 alias /srv/yourplanner/media_production/;

        }


        # YourPlanner Django App

        location /yourplanner/ {

                rewrite /yourplanner/(.*) /$1 break;

                proxy_pass http://unix:/run/yourplanner/yourplanner.sock;

                proxy_set_header Host $host;

                proxy_set_header X-Real-IP $remote_addr;

                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

                proxy_set_header X-Forwarded-Proto $scheme;

                proxy_redirect off;

                proxy_connect_timeout 300s;

                proxy_send_timeout 300s;

                proxy_read_timeout 300s;

        }


	#Happy Payments
	location /HappyPayments/ {
		alias /srv/HappyPayments/website/;
		index index.html;		
	}
	
	# Deny access to .htaccess files

        location ~ /\.ht {

                deny all;

        }



} 


