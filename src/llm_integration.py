import requests
import json
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FreeLLM:
    """LLM WRAPPER"""
    def __init__(self, config):
        self.config = config
        self.model = config.OLLAMA_MODEL
        self.base_url = config.OLLAMA_BASE_URL

        # Check if Ollama is available
        self.ollama_available = self._check_ollama()

        if not self.ollama_available:
            logger.warning(f"Ollama not running! Please start with: ollama serve")
            logger.warning(f"Then pull model: ollama pull {self.model}")

    def _check_ollama(self):
        """Check if Ollama is running and has the configured model"""
        try:
            # Plural /api/tags is the correct Ollama endpoint
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                # Key is 'models', not 'model'
                models = response.json().get('models', [])
                model_names = [m.get('name') for m in models]
                logger.info(f"Available models: {model_names}")

                # Check if our model (or tagged version) is available
                model_exists = False
                for name in model_names:
                    if name == self.model or name.startswith(f"{self.model}:") or self.model.startswith(f"{name}:"):
                        model_exists = True
                        break

                if not model_exists:
                    logger.warning(f"Model {self.model} not found. Pull it with: ollama pull {self.model}")
                else:
                    logger.info(f"✅ {self.model} is available!")
                return True
            return False
        except Exception as e:
            logger.debug(f"Ollama connection check failed: {e}")
            return False


    def generate_answer(self, question: str, context_chunk: List[str]) -> str:
        """
        Generate ans using light weight ollama model
        """
        if not context_chunk:
            return "No relevant context found to answer your question."
        
        context = "\n\n".join(context_chunk[:2])
        if len(context) > 1500:
            context = context[:1500] # small model has 2048 token limit
        
        if self.ollama_available:
            return self._ask_ollama(question, context)

        return self._fallback_response(question, context)
    
    def _ask_ollama(self, question: str, context: str) -> str:
        prompt = f"""Answer the question based on the context.
                    Context: {context}
                    Question: {question}
                    Answer:"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.5,  # Lower temperature for more focused answers
            "max_tokens": 150,   # Keep responses short
            "top_p": 0.9,
            "stop": ["\n\n", "Question:", "Context:"]  # Stop at natural breaks
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                # Clean up common issues
                answer = answer.replace('Answer:', '').strip()
                if not answer:
                    answer = "I found relevant information but couldn't generate a clear answer."
                
                return answer
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return self._fallback_response(question, context)
        
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return self._fallback_response(question, context)

    def stream_answer(self, question: str, context_chunks: List[str]):
        if not context_chunks:
            yield "No relevant context found."
            return
        context = "\n\n".join(context_chunks[:2])
        if len(context) > 1500:
            context = context[:1500] + "..."

        prompt = f"Answer based on context:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "temperature": 0.5,
            "max_tokens": 150
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=30
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get('response', '')
                    if token:
                        yield token
                    if data.get('done', False):
                        break
                        
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield "Error generating response."
        
    def _fallback_response(self, question: str, context: str) -> str:
        """Simple fallback when no LLM is available"""
        # Extract key sentences
        sentences = context.split('.')[:2]
        summary = '. '.join(sentences)
        
        return f"""[Info] Found relevant information:
        {summary}

[Warning] AI model unavailable. Please start Ollama:
1. Open terminal
2. Run: ollama serve
3. Run: ollama pull {self.model}
4. Try again!"""
