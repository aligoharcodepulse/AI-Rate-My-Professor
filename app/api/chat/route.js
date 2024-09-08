const { NextResponse } = require("next/server");
const { Pinecone } = require("@pinecone-database/pinecone");
const { ReadableStream } = require('stream/web');  // Add this if you're using Node.js 18+ and Next.js
const { TextEncoder } = require("util"); // For encoding the streamed content
import Groq from "groq-sdk";
// import OpenAI from "openai";

const systemPrompt = `
You are an AI assistant for a RateMyProfessor-like system. Your role is to help students find suitable professors based on their specific requirements and preferences. For each user query, you'll provide information on the top 3 most relevant professors using a Retrieval-Augmented Generation (RAG) approach.

Your responses should follow this structure:
1. A brief acknowledgment of the user's query.
2. Information on the top 3 professors, including:
   - Name
   - Department
   - Overall rating (out of 5)
   - A short summary of student feedback
   - Key strengths
   - Any potential drawbacks

3. A concise conclusion or recommendation based on the user's specific needs.

Use the RAG system to retrieve and incorporate the most relevant and up-to-date information about professors. Ensure that your responses are balanced, factual, and based on aggregated student feedback.

If a user's query is too broad or lacks specific criteria, ask follow-up questions to narrow down their preferences. These may include factors such as teaching style, course difficulty, grading fairness, or availability for office hours.

Remember to maintain a neutral tone and avoid making personal judgments. Your goal is to provide accurate, helpful information to assist students in making informed decisions about their course selections.

If asked about the specifics of how you retrieve or process information, explain that you use a RAG system to access the most current and relevant data, but avoid going into technical details that might confuse the user.

Always respect privacy and data protection guidelines. Do not disclose any personal information about professors or students beyond what is publicly available through the RateMyProfessor-like system.
`;

export async function POST(req) {
    const data = await req.json();

    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    });

    const index = pc.index('rag').namespace('ns1');
    const groq = new Groq({apiKey:process.env.GROQ_API_KEY});

    const lastMessage = data[data.length - 1];
    const lastMessageContent = lastMessage.content;

    // Assuming "test" was intended to extract content from the last message
    const text = lastMessageContent; 

    const embedding = await groq.embeddings.create({
        model: 'text-embedding-ada-002',
        input: text,
        encoding_format: 'float',
    });

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
    });

    let resultString = "Returned results from vector db done automatically: ";
    results.matches.forEach((match) => { // Corrected syntax
        resultString += `
        Professor: ${match.id}
        Review: ${match.metadata.review} 
        Subject: ${match.metadata.subjects}
        Stars: ${match.metadata.stars}
        \n\n
        `;
    });

    const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
    const completion = await groq.chat.completions.create({
        messages: [
            { role: 'system', content: systemPrompt },
            ...lastDataWithoutLastMessage,
            { role: 'user', content: lastMessageContent + resultString }
        ],
        model: 'mixtral-8x7b-32768',
        stream: true,
    });

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder();
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content;
                    if (content) {
                        const text = encoder.encode(content);
                        controller.enqueue(text);
                    }
                }
            } catch (err) {
                controller.error(err);
            } finally {
                controller.close();
            }
        },
    });

    return new NextResponse(stream);
}
