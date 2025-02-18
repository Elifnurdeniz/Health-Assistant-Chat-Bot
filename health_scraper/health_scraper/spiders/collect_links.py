import scrapy
from scrapy.linkextractors import LinkExtractor
import json
import os

class BlogSpider(scrapy.Spider):
    name = "collect_links"
    start_urls = ["https://www.health.harvard.edu/blog"]


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_file = "/app/data/links.json"

        # Ensure the output file is either created or cleared when the spider starts
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        with open(self.output_file, "w") as f:
            json.dump([], f)  # Initialize as an empty JSON array

    def parse(self, response):
        # Use LinkExtractor to extract all links starting with "/blog"
        link_extractor = LinkExtractor(restrict_css='div.px-6.py-10.md\\:py-12.md\\:px-10.xl\\:p-20', allow=r'/blog')

        # Extract blog links but skip pagination links
        links = link_extractor.extract_links(response)
        for link in links:
            if "?page=" not in link.url:  # Skip links with "?page="
                self.save_link(link.url)
                # yield {
                #     'url': link.url
                # }

        # Find the "Next" page link and follow it
        next_page = response.css('a[rel="next"]::attr(href)').get()
        if next_page:
            yield scrapy.Request(url=response.urljoin(next_page), callback=self.parse)

    def save_link(self, url):
        # Append the link to the output file
        with open(self.output_file, "r+") as f:
            data = json.load(f)  # Load the current list of links
            data.append({"url": url})  # Append the new link
            f.seek(0)  # Move the file pointer to the start of the file
            json.dump(data, f, indent=4)  # Write the updated data with indentation
            f.truncate()  # Remove any leftover data if the new JSON is shorter
