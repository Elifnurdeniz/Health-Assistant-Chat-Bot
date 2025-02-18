FROM python:3.8-slim

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

#WORKDIR /app/health_scraper/spiders

CMD ["scrapy", "crawl", "collect_links.py"]