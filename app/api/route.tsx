import { NextRequest, NextResponse } from "next/server";
import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";
import { streamText } from "ai";
import { AIMessage, ChatMessage, HumanMessage } from "@langchain/core/messages";
import { OpenAIEmbeddings } from "@langchain/openai";
import { UpstashVectorStore } from "@/app/vectorstore/UpstashVectorStore";
import { openai } from '@ai-sdk/openai';

export const runtime = "edge";

const redis = Redis.fromEnv();
const ratelimit = new Ratelimit({
    redis: redis,
    limiter: Ratelimit.slidingWindow(1, "10 s"),
});

const convertVercelMessageToLangChainMessage = (message: { role: string; content: string }) => {
    if (message.role === "user") {
        return new HumanMessage(message.content);
    } else if (message.role === "assistant") {
        return new AIMessage(message.content);
    } else {
        return new ChatMessage(message.content, message.role);
    }
};

export async function POST(req: NextRequest) {
    try {
        const ip = req.headers.get("x-forwarded-for") ?? "127.0.0.1";
        const { success } = await ratelimit.limit(ip);

        if (!success) {
            const customString =
                "Oops! It seems you've reached the rate limit. Please try again later.";

            return NextResponse.json({ error: customString }, { status: 429 });
            //return new StreamingTextResponse(transformStream);
        }

        const body = await req.json();
        const messages = (body.messages ?? []).filter(
            (message: { role: string; content: string }) => message.role === "user" || message.role === "assistant"
        );
        const previousMessages = messages.slice(0, -1).map(convertVercelMessageToLangChainMessage);
        const currentMessageContent = messages[messages.length - 1].content;

        const model = openai('gpt-4o-mini');

        const vectorstore = new UpstashVectorStore(new OpenAIEmbeddings());
        const documents = await vectorstore.similaritySearch(currentMessageContent, 6);
        const context = (documents.map((doc) => doc.pageContent)).join("\n");

        const AGENT_SYSTEM_TEMPLATE = `
   You are an artificial intelligence assistant named HealthAssistant, providing systematic and data-driven health information.

      Begin your answers with a greeting and end with a relevant health tip.

      Your responses should be precise and factual, with an emphasis on using the context provided and providing urls from the context all the time.

      Don't repeat yourself in responses, and if an answer is unavailable in the retrieved content, state that you don't know.

      Now, answer the message below:
      ${currentMessageContent}

        Based on the context below:
        ${context}

        And the previous messages:
        ${previousMessages.map((message: ChatMessage) => message.content).join("\n")}
    `;
    console.log(AGENT_SYSTEM_TEMPLATE);
        const result = await streamText({
            model: model,
            prompt: AGENT_SYSTEM_TEMPLATE,
        })

        return result.toDataStreamResponse();

    } catch (e) {
        if (e instanceof Error) {
            console.error(e.message);
        } else {
            console.error(String(e));
        }
        return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
    }
}
