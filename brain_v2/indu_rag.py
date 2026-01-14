"""
INDU RAG Agent - Conversational AI for SSi Medical Robotics
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


class InduAgent:
    """INDU - Humanoid robot assistant for SS Innovations

    HYBRID Architecture:
    1. Conversation memory - LLM sees full chat history
    2. RAG retrieval - Augments with relevant knowledge base facts
    3. Both work together for context-aware, factual responses
    """

    def __init__(
        self,
        knowledge_base_path: str = "indu_knowledge_base.txt",
        system_prompt_path: str = "indu_system_prompt.txt",
        model_name: str = "llama3.1:8b",
        vector_store_path: str = "indu_vectorstore",
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

        print("✓ INDU Agent initialized (HYBRID mode: LLM memory + RAG)")

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
            formatted.append(f"INDU: {assistant_msg}")
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
        """Send a query to INDU and get response with conversation memory + RAG"""
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
            continuation_instruction = ""
            if is_continuation:
                continuation_instruction = "\n\n[IMPORTANT: This is NOT the first message. You have already introduced yourself. DO NOT say 'I'm Indu' or 'I'm a humanoid robot' again. Just answer the question directly.]"

            # 4. Build the full prompt - HYBRID: System + History + RAG + Query
            prompt = f"""{self.system_prompt}{continuation_instruction}

CONVERSATION HISTORY:
{chat_history if chat_history else "(This is the start of the conversation)"}

RELEVANT KNOWLEDGE BASE FACTS:
{rag_context if rag_context else "(No specific facts retrieved)"}

USER'S CURRENT MESSAGE: {user_query}

CRITICAL INSTRUCTIONS:
- If user says "yes", "go ahead", "please do", "tell me", "explain" - DO NOT ask permission again, just provide the answer
- Read the conversation history above - if you already asked something and user agreed, proceed with the answer
- Follow the system prompt for response length and style
- DO NOT re-introduce yourself after the first message

RESPONSE:"""

            # Debug mode
            if self.debug_mode:
                print("\n" + "="*60)
                print("DEBUG: Full Prompt")
                print("="*60)
                print(f"Chat History Length: {len(self.conversation_history)}")
                print(f"RAG Context Length: {len(rag_context)}")
                print(f"Search Query: {search_query}")
                print("="*60 + "\n")

            # 5. Call LLM with combined prompt
            response = self.llm.invoke(prompt).strip()

            # Handle empty responses
            if not response or len(response) < 5:
                query_lower = user_query.lower()
                if any(word in query_lower for word in ['namaste', 'namaskar']):
                    response = "Namaste! I'm Indu from SSi. How may I assist you today?"
                elif any(word in query_lower for word in ['hello', 'hi', 'hey', 'how are']):
                    response = "Hello! I'm Indu, SSi's humanoid robot. How can I help you today?"
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
    """Interactive CLI for INDU agent"""
    print("=" * 60)
    print("INDU - SS Innovations Humanoid Robot Assistant")
    print("HYBRID Mode: LLM Memory + RAG Knowledge")
    print("=" * 60)
    print()

    agent = InduAgent(debug_mode=True)

    print()
    print("Chat with INDU! Commands: 'exit', 'rebuild', 'clear', 'debug on/off'")
    print("-" * 60)
    print()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("INDU: Goodbye! Happy to help with SSi surgical robotics anytime.")
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
        print(f"INDU: {response}")
        print()


if __name__ == "__main__":
    main()
