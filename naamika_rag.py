"""
NAAMIKA RAG Agent - Conversational AI for SSi Medical Robotics
Uses Ollama + LangChain + FAISS with HYBRID approach:
- LLM maintains conversation memory (like chat)
- RAG augments with knowledge base facts
"""

import os
from typing import List, Tuple
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class NaamikaAgent:
    """NAAMIKA - Humanoid robot assistant for SS Innovations

    HYBRID Architecture:
    1. Conversation memory - LLM sees full chat history
    2. RAG retrieval - Augments with relevant knowledge base facts
    3. Both work together for context-aware, factual responses
    """

    def __init__(
        self,
        knowledge_base_path: str = "naamika_knowledge_base.txt",
        system_prompt_path: str = "naamika_system_prompt.txt",
        model_name: str = "naamika:v1",  # Custom model with baked-in system prompt
        vector_store_path: str = "naamika_vectorstore",
        ollama_host: str = "127.0.0.1",
        ollama_port: int = 11434,
        temperature: float = 0.2,
        debug_mode: bool = False,
        max_history: int = 10
    ):
        self.knowledge_base_path = knowledge_base_path
        self.system_prompt_path = system_prompt_path
        self.model_name = model_name
        self.vector_store_path = vector_store_path
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.temperature = temperature
        self.debug_mode = debug_mode
        self.max_history = max_history

        # Conversation history for context continuity
        self.conversation_history: List[Tuple[str, str]] = []  # [(user, assistant), ...]

        # Load system prompt (rules)
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()

        # Initialize embeddings
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Initialize or load vector store
        if os.path.exists(vector_store_path):
            print("Loading existing vector store...")
            try:
                self.vectorstore = FAISS.load_local(
                    vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading vector store: {e}")
                print("Creating new vector store...")
                self.vectorstore = self._create_vectorstore()
                self.vectorstore.save_local(vector_store_path)
        else:
            print("Creating new vector store from knowledge base...")
            self.vectorstore = self._create_vectorstore()
            self.vectorstore.save_local(vector_store_path)
            print(f"Vector store saved to {vector_store_path}")

        # Initialize retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "fetch_k": 20}
        )

        # Initialize LLM
        ollama_url = f"http://{self.ollama_host}:{self.ollama_port}"
        print(f"Initializing Ollama with {model_name} @ {ollama_url}...")
        self.llm = OllamaLLM(
            model=model_name,
            base_url=ollama_url,
            temperature=self.temperature,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1,
            num_ctx=4096,
        )

        print("✓ NAAMIKA Agent initialized (HYBRID mode: LLM memory + RAG)")

    def _create_vectorstore(self) -> FAISS:
        """Create vector store from knowledge base"""
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            text = f.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", "- ", "  "],
            length_function=len
        )

        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        print(f"Created {len(documents)} document chunks")

        return FAISS.from_documents(documents, self.embeddings)

    def _format_chat_history(self) -> str:
        """Format conversation history for the prompt"""
        if not self.conversation_history:
            return ""

        formatted = []
        for user_msg, assistant_msg in self.conversation_history[-self.max_history:]:
            formatted.append(f"User: {user_msg}")
            formatted.append(f"NAAMIKA: {assistant_msg}")
        return "\n".join(formatted)

    def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant context from knowledge base"""
        docs = self.retriever.invoke(query)
        if not docs:
            return ""

        context_parts = []
        for doc in docs:
            context_parts.append(doc.page_content)
        return "\n\n".join(context_parts)

    def chat(self, user_query: str) -> str:
        """Send a query to NAAMIKA and get response with conversation memory + RAG"""
        try:
            # Build search query - include previous topic for better retrieval
            search_query = user_query
            if self.conversation_history:
                # Add previous topic to search for better RAG retrieval
                last_user, last_assistant = self.conversation_history[-1]
                search_query = f"{last_user} {user_query}"

            # 1. Retrieve relevant context from knowledge base (RAG)
            rag_context = self._retrieve_context(search_query)

            # 2. Format conversation history (Memory)
            chat_history = self._format_chat_history()

            # 3. Check if this is a continuation (to prevent re-introductions)
            is_continuation = len(self.conversation_history) > 0
            continuation_note = "[This is a continuation - do not re-introduce yourself]" if is_continuation else ""

            # 4. Build the prompt - System prompt is BAKED into naamika:v1 model
            # Only send: conversation history + RAG context + user query
            prompt = f"""CONVERSATION HISTORY:
{chat_history if chat_history else "(Start of conversation)"}

RELEVANT KNOWLEDGE BASE FACTS:
{rag_context if rag_context else "(No specific facts)"}

USER: {user_query}
{continuation_note}

RESPONSE:"""

            # Debug mode
            if self.debug_mode:
                print("\n" + "="*60)
                print("DEBUG: Prompt (system prompt baked into model)")
                print("="*60)
                print(f"Model: {self.model_name}")
                print(f"Chat History: {len(self.conversation_history)} turns")
                print(f"RAG Context: {len(rag_context)} chars")
                print(f"Prompt size: {len(prompt)} chars (~{len(prompt)//4} tokens)")
                print("="*60 + "\n")

            # 5. Call LLM with combined prompt
            response = self.llm.invoke(prompt).strip()

            # Handle empty responses
            if not response or len(response) < 5:
                query_lower = user_query.lower()
                if any(word in query_lower for word in ['namaste', 'namaskar']):
                    response = "Namaste! I'm Naamika from SSi. How may I assist you today?"
                elif any(word in query_lower for word in ['hello', 'hi', 'hey', 'how are']):
                    response = "Hello! I'm Naamika, SSi's humanoid robot. How can I help you today?"
                elif any(word in query_lower for word in ['joke', 'funny', 'laugh']):
                    response = "Why did the robot go to medical school? To get better at operating systems!"
                else:
                    response = "I'm specialized in SSi medical robotics. What would you like to know?"

            # 6. Save to conversation history
            self.conversation_history.append((user_query, response))

            # Trim history if too long
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]

            return response

        except Exception as e:
            return f"I encountered an error: {str(e)}"

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def rebuild_vectorstore(self):
        """Rebuild vector store from knowledge base"""
        print("Rebuilding vector store...")
        self.vectorstore = self._create_vectorstore()
        self.vectorstore.save_local(self.vector_store_path)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        print("✓ Vector store rebuilt!")


def main():
    """Interactive CLI for NAAMIKA agent"""
    print("=" * 60)
    print("NAAMIKA - SS Innovations Humanoid Robot Assistant")
    print("HYBRID Mode: LLM Memory + RAG Knowledge")
    print("=" * 60)
    print()

    agent = NaamikaAgent(debug_mode=True)

    print()
    print("Chat with NAAMIKA! Commands: 'exit', 'rebuild', 'clear', 'debug on/off'")
    print("-" * 60)
    print()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("NAAMIKA: Goodbye! Happy to help with SSi surgical robotics anytime.")
            break

        if user_input.lower() == 'rebuild':
            agent.rebuild_vectorstore()
            continue

        if user_input.lower() == 'clear':
            agent.clear_history()
            print("Conversation history cleared!")
            continue

        if user_input.lower() in ['debug on', 'debug']:
            agent.debug_mode = True
            print("Debug mode enabled")
            continue

        if user_input.lower() == 'debug off':
            agent.debug_mode = False
            print("Debug mode disabled")
            continue

        response = agent.chat(user_input)
        print(f"NAAMIKA: {response}")
        print()


if __name__ == "__main__":
    main()
