import dotenv from 'dotenv';
import OpenAI from 'openai';
import {Pinecone} from '@pinecone-database/pinecone';
import axios from 'axios';

dotenv.config();

// Initialize OpenAI
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

// Initialize Pinecone
const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
});

// Function to generate embeddings
async function getEmbedding(text){
    const response = await openai.embeddings.create({
        input: text,
        model: "text-embedding-ada-002",
    });
    return response.data.data[0].embedding;
}

// Sample data
// const response = await axios.get(process.env.DATA_URL);
// const data = response.data;
// console.log("data",data);

const data = [
    { id: "1", text: "How to fix a car engine" },
    { id: "2", text: "Best practices for car maintenance" },
    { id: "3", text: "Top 10 car repair tips" },
  ];

// Main function
async function main(){
    // Step 1: Generate embeddings for the data
    for(const item of data){
        item.embedding = await getEmbedding(item.text);
    }

    // Step 2: Store embeddings in Pinecone
    const index = pinecone.index('semantic-search');
    await index.upsert(data.map(item => ({
        id: item.id,
        values: item.embedding,
        metadata: {text: item.text},
    })));
    console.log("Data uploaded to Pinecone!");

    // Step 3: Query Pinecone for similar vectors
    const query = "How to repair a car";
    const queryEmbedding = await getEmbedding(query);

    const results = await index.query({
        vector: queryEmbedding,
        topK: 3,
        includeMetadata: true,
    });

    console.log("Results: ", results);
    results.matches.forEach(match => {
        console.log(`ID: ${match.id}, Text: ${match.metadata.text}, Score: ${match.score}`);
    });
}

main().catch(console.error);