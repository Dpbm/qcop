FROM postgres:17.2-alpine3.21

COPY db-entrypoint.sh /docker-entrypoint-initdb.d/entrypoint.sh

RUN chmod +x /docker-entrypoint-initdb.d/entrypoint.sh
