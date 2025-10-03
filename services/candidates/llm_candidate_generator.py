"""
LLM Candidate Generation Service

Generates entity, relation, and topic suggestions using LLM with domain-specific prompts.
Supports both OpenAI API and local models via Ollama.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import asyncio
from datetime import datetime

# For OpenAI support
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# For local model support (Ollama)
try:
    import requests
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

logger = logging.getLogger(__name__)

@dataclass
class EntityCandidate:
    """Entity candidate from LLM"""
    cid: int
    text: str
    label: str
    start: int
    end: int
    confidence: float

@dataclass
class RelationCandidate:
    """Relation candidate from LLM"""
    rid: int
    head_cid: int
    tail_cid: int
    label: str
    evidence: Optional[str] = None
    confidence: float = 0.0

@dataclass
class TopicCandidate:
    """Topic candidate from LLM"""
    topic_id: str
    label: str
    score: float
    keywords: List[str] = None

class LLMCandidateGenerator:
    """
    Generate annotation candidates using LLM.
    
    Supports:
    - OpenAI GPT models (gpt-4, gpt-3.5-turbo)
    - Ollama local models (llama2, mistral, etc)
    - Cached responses to reduce API calls
    - Batch processing for efficiency
    """
    
    def __init__(self, 
                 provider: str = "openai",
                 model: str = "gpt-4o-mini", 
                 api_key: Optional[str] = None,
                 temperature: float = 0.1,
                 cache_dir: Optional[Path] = None):
        """
        Initialize the candidate generator.
        
        Args:
            provider: "openai" or "ollama"
            model: Model name
            api_key: API key for OpenAI
            temperature: Sampling temperature
            cache_dir: Directory for caching responses
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load prompts
        prompt_file = Path(__file__).parent.parent.parent / "shared/prompts/entity_extraction_prompts.json"
        with open(prompt_file, 'r') as f:
            self.prompts = json.load(f)
        
        # Setup provider
        if provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai package not installed")
            if not api_key:
                raise ValueError("OpenAI API key required")
            self.client = openai.OpenAI(api_key=api_key)
        elif provider == "ollama":
            if not HAS_OLLAMA:
                raise ImportError("requests package not installed")
            self.ollama_url = "http://localhost:11434/api/generate"
        else:
            raise ValueError(f"Unknown provider: {provider}")
            
        # Setup cache
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized {provider} generator with model {model}")
    
    def _get_cache_key(self, text: str, task: str) -> str:
        """Generate cache key for a request"""
        import hashlib
        content = f"{self.model}:{task}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached response"""
        if not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, response: Dict):
        """Save response to cache"""
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(response, f)
    
    async def _call_openai(self, prompt: str, system_prompt: str) -> Dict:
        """Call OpenAI API with error handling and retries"""
        from utils.error_handling import retry_with_backoff, RetryConfig, error_handler
        
        @retry_with_backoff(config=RetryConfig(max_attempts=3, initial_delay=1.0))
        async def _make_api_call():
            try:
                # Use circuit breaker for OpenAI calls
                circuit_breaker = error_handler.get_circuit_breaker(
                    "openai_api",
                    failure_threshold=5,
                    recovery_timeout=60.0
                )
                
                async def api_call():
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=500,
                        timeout=30.0  # Add timeout
                    )
                    
                    content = response.choices[0].message.content
                    if not content:
                        raise ValueError("Empty response from OpenAI API")
                    
                    # Parse JSON from response
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse OpenAI JSON response: {content[:100]}...")
                        # Try to extract JSON from response
                        import re
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            return json.loads(json_match.group(0))
                        raise ValueError(f"Invalid JSON in OpenAI response: {e}")
                
                return await circuit_breaker.async_call(api_call)
                
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                # Return empty structure instead of failing completely
                raise
        
        try:
            return await _make_api_call()
        except Exception as e:
            logger.error(f"All OpenAI API attempts failed: {e}")
            # Return empty structure to allow graceful degradation
            return {"entities": [], "relations": [], "topics": []}
    
    def _call_ollama(self, prompt: str, system_prompt: str) -> Dict:
        """Call Ollama local model with error handling"""
        from utils.error_handling import retry_with_backoff, RetryConfig, error_handler
        
        @retry_with_backoff(config=RetryConfig(max_attempts=2, initial_delay=0.5))
        def _make_ollama_call():
            circuit_breaker = error_handler.get_circuit_breaker(
                "ollama_api",
                failure_threshold=3,
                recovery_timeout=30.0
            )
            
            def api_call():
                full_prompt = f"{system_prompt}\n\n{prompt}"
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "temperature": self.temperature,
                        "stream": False,
                        "format": "json"
                    },
                    timeout=60.0  # Add timeout
                )
                
                response.raise_for_status()  # Raise exception for HTTP errors
                
                result = response.json()
                response_text = result.get("response", "{}")
                
                if not response_text:
                    raise ValueError("Empty response from Ollama")
                
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse Ollama JSON: {response_text[:100]}...")
                    raise ValueError(f"Invalid JSON from Ollama: {e}")
            
            return circuit_breaker.call(api_call)
        
        try:
            return _make_ollama_call()
        except Exception as e:
            logger.error(f"All Ollama attempts failed: {e}")
            # Return empty structure for graceful degradation
            return {"entities": [], "relations": [], "topics": []}
    
    async def extract_entities(self, sentence: str) -> List[EntityCandidate]:
        """
        Extract entity candidates from a sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            List of entity candidates
        """
        # Check cache
        cache_key = self._get_cache_key(sentence, "entities")
        cached = self._load_from_cache(cache_key)
        if cached:
            return [EntityCandidate(**e) for e in cached.get("entities", [])]
        
        # Build prompt
        prompt_config = self.prompts["entity_extraction"]
        prompt = prompt_config["main_prompt"].format(sentence=sentence)
        
        # Add few-shot examples
        if prompt_config.get("few_shot_examples"):
            examples_text = "\n\nExamples:\n"
            for ex in prompt_config["few_shot_examples"][:2]:
                examples_text += f"Input: {ex['sentence']}\n"
                examples_text += f"Output: {json.dumps(ex['output'])}\n\n"
            prompt = examples_text + prompt
        
        # Call LLM
        if self.provider == "openai":
            result = await self._call_openai(prompt, prompt_config["system_prompt"])
        else:
            result = self._call_ollama(prompt, prompt_config["system_prompt"])
        
        # Parse results with span validation
        entities = []
        for i, entity_dict in enumerate(result.get("entities", [])):
            try:
                # Validate span matches text
                start = entity_dict["start"]
                end = entity_dict["end"]
                expected_text = entity_dict["text"]
                
                # Perform span validation
                validated_start, validated_end = self._validate_entity_span(
                    sentence, expected_text, start, end
                )
                
                # Only add entity if span validation passed
                if validated_start is not None and validated_end is not None:
                    entities.append(EntityCandidate(
                        cid=i,
                        text=sentence[validated_start:validated_end],  # Use actual text from validated span
                        label=entity_dict["label"],
                        start=validated_start,
                        end=validated_end,
                        confidence=entity_dict.get("confidence", 0.8)
                    ))
                else:
                    logger.warning(f"Failed span validation for entity: '{expected_text}' at position {start}-{end}")
                    
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse entity: {e}")
        
        # Cache results
        if entities:
            self._save_to_cache(cache_key, {"entities": [asdict(e) for e in entities]})
        
        return entities
    
    async def extract_relations(self, 
                                sentence: str, 
                                entities: List[EntityCandidate]) -> List[RelationCandidate]:
        """
        Extract relation candidates given entities.
        
        Args:
            sentence: Input sentence
            entities: List of entities in the sentence
            
        Returns:
            List of relation candidates
        """
        if len(entities) < 2:
            return []
        
        # Prepare entity list for prompt
        entity_list = [
            {"index": i, "text": e.text, "label": e.label}
            for i, e in enumerate(entities)
        ]
        
        # Build prompt
        prompt_config = self.prompts["relation_extraction"]
        prompt = prompt_config["main_prompt"].format(
            sentence=sentence,
            entities=json.dumps(entity_list, indent=2)
        )
        
        # Call LLM
        if self.provider == "openai":
            result = await self._call_openai(prompt, prompt_config["system_prompt"])
        else:
            result = self._call_ollama(prompt, prompt_config["system_prompt"])
        
        # Parse results
        relations = []
        for i, rel_dict in enumerate(result.get("relations", [])):
            try:
                relations.append(RelationCandidate(
                    rid=i,
                    head_cid=rel_dict["head"],
                    tail_cid=rel_dict["tail"],
                    label=rel_dict["label"],
                    evidence=rel_dict.get("evidence"),
                    confidence=rel_dict.get("confidence", 0.7)
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse relation: {e}")
        
        return relations
    
    async def suggest_topics(self, 
                            text: str, 
                            title: Optional[str] = None) -> List[TopicCandidate]:
        """
        Suggest topics for a text segment.
        
        Args:
            text: Input text (sentence or paragraph)
            title: Optional document title for context
            
        Returns:
            List of topic candidates
        """
        # Build prompt
        prompt_config = self.prompts["topic_suggestion"]
        prompt = prompt_config["main_prompt"].format(
            title=title or "Unknown",
            text=text[:500]  # Limit text length
        )
        
        # Call LLM
        if self.provider == "openai":
            result = await self._call_openai(prompt, prompt_config["system_prompt"])
        else:
            result = self._call_ollama(prompt, prompt_config["system_prompt"])
        
        # Parse results
        topics = []
        for topic_dict in result.get("topics", []):
            try:
                topics.append(TopicCandidate(
                    topic_id=topic_dict["topic_id"],
                    label=topic_dict.get("label", topic_dict["topic_id"]),
                    score=topic_dict.get("score", 0.5),
                    keywords=topic_dict.get("keywords", [])
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse topic: {e}")
        
        return topics
    
    async def process_sentence(self, 
                               doc_id: str,
                               sent_id: str,
                               sentence: str,
                               title: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a complete sentence to generate all candidates.
        
        Args:
            doc_id: Document ID
            sent_id: Sentence ID
            sentence: Sentence text
            title: Optional document title
            
        Returns:
            Complete candidate dictionary
        """
        start_time = time.time()
        
        # Extract entities
        entities = await self.extract_entities(sentence)
        
        # Extract relations if entities found
        relations = []
        if entities:
            relations = await self.extract_relations(sentence, entities)
        
        # Suggest topics
        topics = await self.suggest_topics(sentence, title)
        
        # Build result
        result = {
            "doc_id": doc_id,
            "sent_id": sent_id,
            "candidates": {
                "entities": [asdict(e) for e in entities],
                "relations": [asdict(r) for r in relations],
                "topics": [asdict(t) for t in topics]
            },
            "model_info": {
                "model": self.model,
                "provider": self.provider,
                "temperature": self.temperature
            },
            "timestamp": datetime.now().isoformat(),
            "processing_time": time.time() - start_time
        }
        
        return result
    
    async def process_batch(self, 
                           sentences: List[Dict[str, str]],
                           batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Process multiple sentences in batches.
        
        Args:
            sentences: List of sentence dicts with doc_id, sent_id, text, title
            batch_size: Number of sentences to process concurrently
            
        Returns:
            List of candidate dictionaries
        """
        results = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            # Process batch concurrently
            tasks = [
                self.process_sentence(
                    s["doc_id"],
                    s["sent_id"],
                    s["text"],
                    s.get("title")
                )
                for s in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            logger.info(f"Processed {len(results)}/{len(sentences)} sentences")
        
        return results
    
    def _validate_entity_span(self, 
                             sentence: str, 
                             expected_text: str, 
                             start: int, 
                             end: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Validate and correct entity span positions.
        
        Args:
            sentence: Full sentence text
            expected_text: Expected entity text from LLM
            start: Proposed start position
            end: Proposed end position
            
        Returns:
            Tuple of (validated_start, validated_end) or (None, None) if validation fails
        """
        if not expected_text or not sentence:
            return None, None
            
        # Normalize whitespace for comparison
        expected_clean = expected_text.strip()
        
        # Check if the proposed span matches exactly
        if 0 <= start < end <= len(sentence):
            actual_text = sentence[start:end].strip()
            if actual_text.lower() == expected_clean.lower():
                return start, end
        
        # Try to find the text in the sentence (fuzzy matching)
        expected_lower = expected_clean.lower()
        sentence_lower = sentence.lower()
        
        # Look for exact substring match
        exact_start = sentence_lower.find(expected_lower)
        if exact_start != -1:
            exact_end = exact_start + len(expected_lower)
            return exact_start, exact_end
        
        # Try partial matches (for cases where LLM extracted part of a word)
        words = expected_clean.split()
        if len(words) == 1:
            # Single word - look for word boundaries
            import re
            pattern = r'\b' + re.escape(expected_lower) + r'\b'
            match = re.search(pattern, sentence_lower)
            if match:
                return match.start(), match.end()
        
        # Try to find the first and last words of multi-word entities
        if len(words) > 1:
            first_word = words[0].lower()
            last_word = words[-1].lower()
            
            # Find first word
            first_start = sentence_lower.find(first_word)
            if first_start != -1:
                # Look for last word after first word
                search_start = first_start + len(first_word)
                last_start = sentence_lower.find(last_word, search_start)
                if last_start != -1:
                    last_end = last_start + len(last_word)
                    # Check if the span makes sense (not too far apart)
                    if last_end - first_start < len(expected_clean) * 2:
                        return first_start, last_end
        
        # Fuzzy matching for small differences (typos, case variations)
        if len(expected_clean) >= 3:
            for i in range(len(sentence) - len(expected_clean) + 1):
                candidate = sentence[i:i + len(expected_clean)]
                # Simple similarity check
                if self._text_similarity(expected_clean.lower(), candidate.lower()) > 0.8:
                    return i, i + len(expected_clean)
        
        # Last resort: return None to indicate validation failure
        logger.debug(f"Could not validate span for '{expected_text}' in sentence: '{sentence[:50]}...'")
        return None, None
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity score.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
            
        if text1 == text2:
            return 1.0
        
        # Simple character-based similarity
        if len(text1) != len(text2):
            return 0.0
            
        matches = sum(1 for a, b in zip(text1, text2) if a == b)
        return matches / len(text1)


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LLM candidates")
    parser.add_argument("--provider", default="openai", choices=["openai", "ollama"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--sentence", help="Test sentence")
    parser.add_argument("--input-file", help="JSONL file with sentences")
    parser.add_argument("--output-file", help="Output JSONL file")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = LLMCandidateGenerator(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        cache_dir=Path("../../data/candidates/.cache")
    )
    
    async def main():
        if args.sentence:
            # Test single sentence
            result = await generator.process_sentence(
                "test_doc",
                "test_sent",
                args.sentence
            )
            print(json.dumps(result, indent=2))
            
        elif args.input_file:
            # Process file
            sentences = []
            with open(args.input_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    sentences.append(data)
            
            results = await generator.process_batch(sentences)
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    for result in results:
                        f.write(json.dumps(result) + "\n")
                print(f"Wrote {len(results)} results to {args.output_file}")
            else:
                for result in results:
                    print(json.dumps(result))
    
    asyncio.run(main())