# Health Assistant Chatbot 
## Building a Modern Health Assistant with Next.js and Upstash
## Introduction
In this tutorial, we'll dive into how we built a modern Health Assistant application using Next.js and Upstash. Health Assistant project is a RAG chat app that is trained on health related data. Our goal is to create an interactive platform that uses AI to provide insights and advices to users for their health related questions.
## Tech Stack
- Data Collection: [scrapy crawler](https://scrapy.org/)
- Application: [Next.js](https://nextjs.org/)
- Vector Database: [Upstash]([https://upstash.com/](https://upstash.com/docs/vector/overall/getstarted))
- LLM Orchestration: [Langchain.js](https://js.langchain.com)
- Generative Model: [OpenAI](https://openai.com/)
- Middleware: [Vercel AI](https://vercel.com/ai)
- Rate Limiting: [Upstash](https://upstash.com/docs/redis/sdks/ratelimit-ts/overview)

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

### What is a RAG Chatbot?
Retrieval-Augmented Generation (RAG) is an AI model that optimizes its responses by including information retrieved from a knowledge base based on the user input. It has 2 main phases:
1. Retrieval Phase: When a user asks a question, the chatbot first searches through a large database of previously stored knowledge bases to find relevant information. Vector databases provides the opportunity for similarity searches and retrieves related data.
2. Generation Phase: The relevant information obtained during the retrieval phase is then fed into a generative AI model. Once the response is created, it is displayed to the user in real-time, using the streaming properties.

### Data Collection And Storage
Data collection of this project is handled by Scrapy, an open-source and powerful web-crawling framework written in Python. We start by [initializing a Scrapy project](https://docs.scrapy.org/en/latest/intro/tutorial.html#creating-a-project) and [customizing our spider](https://github.com/Elifnurdeniz/Health-Assistant-Chat-Bot/tree/main/health_scraper/health_scraper/spiders) based on the data source. The `parse_page` function in the [spider](https://github.com/Elifnurdeniz/Health-Assistant-Chat-Bot/blob/main/health_scraper/health_scraper/spiders/get_content_and_store.py) collects selected sections' data, splits them into chunks, generates vector embeddings for them, and uploads those embeddings to the Upstash Vector Database. 

</br>

To run the crawler, follow these steps:
- Clone the repository: `git clone https://github.com/YOUR_GITHUB_ACCOUNT/Health-Assistant-Chat-Bot`
- Create a .env file in the `health_scraper` folder as in the [example](https://github.com/Elifnurdeniz/Health-Assistant-Chat-Bot/blob/main/health_scraper/.env.example).
If you don't already have an Upstash Vector Database, create one [here](https://console.upstash.com/vector) and set 1536 as the vector dimensions. Similarly, if you dont have an Open AI key, creare one [here](https://openai.com/index/openai-api/).
- Then, simply run `docker compose up`. This will create a container running your crawler. If you want to run the crawlers in this project, two crawlers should be ran seperately, in order of `docker compose up collect links` and `docker compose up fetch_content`.

</br>

To customize your crawler, you can change the code segment that handles data extractions as in this example `elements = response.xpath("//div[contains(@class, 'content-repository-content')]//p | //div[contains(@class, 'content-repository-content')]//li")`.

</br>

❗ Note: Running crawler may take time. To see the progress, you can check check the logs or monitor your vector database from your Upstash account. 

![image](https://github.com/user-attachments/assets/6c3caae2-189e-4305-9f78-4a15f1038149)

### Data Retrieval And Response Generation


First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.



## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
