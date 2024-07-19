import os
import asyncio
from dotenv import load_dotenv
from typing import List, Any, AsyncGenerator
from index import PDFProcessor
import time
from openai import OpenAI

class RAGChatbot:
    def __init__(self, model_name: str = "llama-3-8b-chat@fireworks-ai"):
        
        
        self.client = OpenAI(
            base_url="https://api.unify.ai/v0/",
            api_key="KHVDuvZfTziakXR4mxJdQZ5btBPdLkDKzJCutkfqg4I="
        )
        self.pdf_processor = PDFProcessor()
        self.model_name = model_name

    async def load_pdf(self, pdf_path: str, index_name: str = "pdf_index"):
        await asyncio.to_thread(self.pdf_processor.load_and_process_pdf, pdf_path, index_name)

    async def retrieve(self, query: str, num_nodes: int = 3) -> List[Any]:
        if self.pdf_processor.index is None:
            raise ValueError("No index loaded. Please load a PDF first.")
        
        retriever = self.pdf_processor.index.as_retriever(similarity_top_k=num_nodes)
        return await asyncio.to_thread(retriever.retrieve, query)

    async def generate_stream(self, query: str, retrieved_nodes: List[Any]) -> AsyncGenerator[str, None]:
        context = "\n".join([str(node.node.get_content()) for node in retrieved_nodes])
        prompt = f"Context:\n{context}\n\nUser Query: {query}\n\nResponse:"
        
        stream = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are ArcAI, an AI assistant. Provide helpful and accurate responses based on the given context."},
                {"role": "user", "content": prompt},
            ],
            model=self.model_name,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                yield content

    async def query(self, user_input: str) -> AsyncGenerator[str, None]:
        print("Processing your query...", end="", flush=True)
        retrieved_nodes = await self.retrieve(user_input)
        print("\rHere's the response:           ")  # Clear the "Processing" message
        
        async for response_chunk in self.generate_stream(user_input, retrieved_nodes):
            yield response_chunk

async def main():
    chatbot = RAGChatbot()

    print("Loading PDF...")
    await chatbot.load_pdf("WarehouseStock&Kitting&jobCards.txt", "my_pdf_index")
    print("PDF loaded successfully!")

    greeting = "Hello! I'm ArcAI, your AI assistant. How can I assist you today?"
    print("ArcAI:", greeting)

    while True:
        user_query = input("You: ")

        if user_query.lower() in ['exit', 'quit', 'bye']:
            break
        
        start_time = time.time()
        print("ArcAI: ", end="", flush=True)
        async for response_chunk in chatbot.query(user_query):
            print(response_chunk, end="", flush=True)
        print()  # New line after the complete response
        end_time = time.time()
        
        print(f"Response time: {end_time - start_time:.2f} seconds")

    farewell = "Goodbye! Have a great day!"
    print("ArcAI:", farewell)

if __name__ == "__main__":
    asyncio.run(main())