import scrapy
import json
import os
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from upstash_vector import Index
from ..utils.upstash_vector_store import UpstashVectorStore
from dotenv import load_dotenv

class ContentSpider(scrapy.Spider):
    name = "get_content_and_store"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open("/app/data/links.json", "r") as f:
            self.links = json.load(f)

        # Initialize Upstash Vector Store
        load_dotenv()

        self.vectorstore = UpstashVectorStore(
            url=os.getenv("UPSTASH_VECTOR_REST_URL"),
            token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
        )
        print(os.getenv("UPSTASH_VECTOR_REST_URL"))

        # Text splitter configuration for handling large texts
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def start_requests(self):
        """
        Initiates the scraping process by iterating through the links in links.json.
        """
        for link in self.links:
            yield scrapy.Request(url=link["url"], callback=self.parse_page)


    def parse_page(self, response):
        """
        Processes each blog page, extracts content, splits it into chunks,
        and stores the data in the Upstash vector database.
        """
        # Extract text content from paragraphs
        elements = response.xpath("//div[contains(@class, 'content-repository-content')]//p | //div[contains(@class, 'content-repository-content')]//li")


        # Combine all text content into a single string
        combined_text = "\n".join([element.xpath("string(.)").get() for element in elements])

        # Check if there is valid content
        if not combined_text.strip():
            print(f"No valid text found on page: {response.url}")
            return

        # Split combined text into chunks
        documents = self.text_splitter.split_text(combined_text)

        # Print the split chunks for debugging (optional)
        #print(f"Split into {len(documents)} chunks: {documents}")

        # Skip if no valid documents
        if len(documents) == 0:
            return

        # Generate unique IDs and add documents to Upstash
        self.vectorstore.add(
            ids=[str(uuid.uuid4())[:8] for _ in documents],
            documents=documents,
            link=response.url,
        )
