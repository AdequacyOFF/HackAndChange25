FROM node:20-alpine AS build

WORKDIR /site
COPY . .

RUN npm ci
RUN npm run build


FROM nginx:alpine

RUN apk add --no-cache openssl
RUN rm /etc/nginx/conf.d/default.conf

COPY --from=build /site/dist /etc/nginx/html
COPY nginx.conf /etc/nginx/conf.d/

RUN mkdir -p /etc/nginx/ssl