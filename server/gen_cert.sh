#!/bin/bash

# Generate valid CA
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 365 -key ca.key -out ca.crt -subj  "/C=US/ST=Texas/L=Austin/O=Test/OU=Test/CN=Root CA"

# Generate valid Server Key/Cert
openssl genrsa -out server.key 4096
openssl req -new -key server.key -out server.csr -subj  "/C=US/ST=Texas/L=Austin/O=Test/OU=Test/CN=server-instance"
openssl x509 -req -days 365 -in server.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out server.crt

# Generate valid Client Key/Cert
openssl genrsa -out client.key 4096
openssl req -new -key client.key -out client.csr -subj "/C=US/ST=Texas/L=Austin/O=Test/OU=Test/CN=client-instance"
openssl x509 -req -days 365 -in client.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out client.crt
