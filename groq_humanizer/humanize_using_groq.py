import os
import re
from typing import List, Optional
from groq import Groq
from dotenv import load_dotenv
import sys
from datetime import datetime

load_dotenv()

def sanitize_unicode_text(text: str) -> str:
    """
    Clean text to remove invalid Unicode surrogate characters that cause encoding errors.
    
    Args:
        text: Input text that may contain problematic Unicode characters
        
    Returns:
        Cleaned text safe for UTF-8 encoding
    """
    if not text:
        return text
    
    # Remove or replace Unicode surrogate characters (U+D800 to U+DFFF)
    # These are invalid in UTF-8 and cause encoding errors
    sanitized = text.encode('utf-8', 'replace').decode('utf-8')
    
    # Remove any remaining problematic characters
    # This regex removes characters that might cause issues
    sanitized = re.sub(r'[\uD800-\uDFFF]', '', sanitized)
    
    # Also remove any null bytes or other control characters that might cause issues
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
    
    return sanitized

class HumanizeTextWithGroq:
    """
    A class to humanize AI-generated text using Groq API.
    Processes text paragraph by paragraph with multiple iterations for better humanization.
    """
    
    # Common AI phrases to avoid during humanization
    common_ai_phrases_to_avoid = [
        "as an AI language model",
        "I'm sorry, but",
        "it is important to note that",
        "in conclusion",
        "here are some key points",
        "delve into",
        "in the realm of",
        "plays a crucial role",
        "gain valuable insights",
        "embark on your journey",
        "seamlessly integrated",
        "innovative solutions",
        "elevate your experience",
        "unlock the potential",
        "fostering a culture of",
        # Most prevalent AI phrases based on 2024-2025 research (50 phrases)
        "today's fast-paced world",
        "notable works include",
        "aims to explore",
        "aligns with",
        "surpassing expectations",
        "tragically",
        "making an impact",
        "research needed to understand",
        "despite facing",
        "expressed excitement",
        "evolving situation",
        "at its core",
        "to put it simply",
        "this underscores the importance of",
        "a key takeaway is",
        "that being said",
        "from a broader perspective",
        "generally speaking",
        "typically",
        "tends to",
        "arguably",
        "to some extent",
        "broadly speaking",
        "shed light on",
        "facilitate",
        "refine",
        "bolster",
        "differentiate",
        "streamline",
        "revolutionize",
        "cutting-edge",
        "game-changing",
        "transformative",
        "seamless integration",
        "excitingly",
        "amazing",
        "remarkable",
        "revolutionize the way",
        "transformative power",
        "groundbreaking advancement",
        "pushing the boundaries",
        "only time will tell",
        "rapid pace of development",
        "bringing us one step closer",
        "paving the way",
        "significantly enhances",
        "aims to democratize",
        "continues to progress rapidly",
        "exciting opportunities",
        "opens up exciting possibilities",
        "unleashing the potential",
        "exploring new frontiers",
        "underscore",
        "pivotal",
        "harness",
        "illuminate",
        "unlock the secrets",
        "unveil the secrets",
        "take a dive into",
        "in today's digital era",
        "in summary",
        "profound",
        "supercharge",
        "evolve",
        "reimagine",
        "navigate",
        "moreover",
        "therefore",
        "alternatively",
        "specifically"
    ]
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile", last_pass: bool = False):
        """
        Initialize the humanizer with Groq API.
        
        Args:
            api_key: Groq API key (if None, will try to get from environment)
            model: Groq model to use for humanization
            last_pass: Whether to perform a final coherence pass on the entire text (default: True)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.last_pass = last_pass
    
    def _detect_paragraphs(self, text: str) -> List[str]:
        """
        Detect paragraphs in the text and ensure minimum chunk sizes.
        Merges small chunks (≤20 words) with neighboring chunks.
        
        Args:
            text: Input text to split into paragraphs
            
        Returns:
            List of paragraph strings with minimum 20 words each
        """
        # Clean up the text first
        text = text.strip()
        
        # Log initial detection
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] 🔍 Starting paragraph detection and chunking...")
        
        # Split by double newlines first (most common paragraph separator)
        initial_paragraphs = re.split(r'\n\s*\n', text)
        
        # If no double newlines found, split by single newlines
        if len(initial_paragraphs) == 1:
            initial_paragraphs = text.split('\n')
        
        # Clean and filter empty paragraphs
        initial_paragraphs = [p.strip() for p in initial_paragraphs if p.strip()]
        
        print(f"[{timestamp}] 📄 Found {len(initial_paragraphs)} initial paragraphs")
        
        # Log paragraph sizes
        for i, para in enumerate(initial_paragraphs):
            word_count = self._count_words(para)
            print(f"[{timestamp}]   Paragraph {i+1}: {word_count} words")
        
        # Split very long paragraphs (more than 500 characters) at sentence boundaries
        split_paragraphs = []
        for para in initial_paragraphs:
            if len(para) > 500:
                # Split by periods followed by space and capital letter (sentence boundaries)
                sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', para)
                
                # Group sentences into chunks (targeting ~300 chars but ensuring word count minimums)
                current_chunk = ""
                for sentence in sentences:
                    test_chunk = current_chunk + sentence + " "
                    
                    # If adding this sentence makes it too long AND current chunk has enough words
                    if (len(test_chunk) > 300 and 
                        current_chunk.strip() and 
                        self._count_words(current_chunk) >= 20):
                        split_paragraphs.append(current_chunk.strip())
                        current_chunk = sentence + " "
                    else:
                        current_chunk = test_chunk
                
                if current_chunk.strip():
                    split_paragraphs.append(current_chunk.strip())
            else:
                split_paragraphs.append(para)
        
        # Now merge small chunks (≤20 words) with neighboring chunks
        final_chunks = self._merge_small_chunks(split_paragraphs)
        
        # Log final chunk information
        print(f"[{timestamp}] ✅ Final chunking complete: {len(final_chunks)} chunks")
        for i, chunk in enumerate(final_chunks):
            word_count = self._count_words(chunk)
            print(f"[{timestamp}]   Chunk {i+1}: {word_count} words")
        
        return final_chunks
    
    def _merge_small_chunks(self, paragraphs: List[str]) -> List[str]:
        """
        Merge chunks with ≤20 words with neighboring chunks.
        
        Args:
            paragraphs: List of paragraph strings
            
        Returns:
            List of merged paragraphs with minimum 20 words each
        """
        if not paragraphs:
            return []
        
        # If only one paragraph, return it regardless of size
        if len(paragraphs) == 1:
            return paragraphs
        
        merged_chunks = []
        i = 0
        
        while i < len(paragraphs):
            current_chunk = paragraphs[i]
            current_word_count = self._count_words(current_chunk)
            
            # If current chunk is too small, try to merge it
            if current_word_count <= 20:
                # Try to merge with next chunk first (if exists)
                if i + 1 < len(paragraphs):
                    next_chunk = paragraphs[i + 1]
                    merged = current_chunk + "\n\n" + next_chunk
                    merged_word_count = self._count_words(merged)
                    
                    # If merged chunk is reasonable size (not too big), merge them
                    if merged_word_count <= 150:  # Reasonable upper limit
                        merged_chunks.append(merged)
                        i += 2  # Skip both chunks as they're now merged
                        continue
                
                # If can't merge with next, try to merge with previous (if exists and it's the last chunk)
                if merged_chunks and i == len(paragraphs) - 1:
                    # Merge with the last chunk in merged_chunks
                    last_chunk = merged_chunks.pop()
                    merged = last_chunk + "\n\n" + current_chunk
                    merged_chunks.append(merged)
                    i += 1
                    continue
                
                # If can't merge, keep it anyway (edge case)
                merged_chunks.append(current_chunk)
                i += 1
            else:
                # Chunk is large enough, keep it as is
                merged_chunks.append(current_chunk)
                i += 1
        
        return merged_chunks
    
    def _count_words(self, text: str) -> int:
        """
        Count words in text, excluding extra whitespace.
        
        Args:
            text: Input text to count words in
            
        Returns:
            Number of words
        """
        return len(text.strip().split())
    
    def _get_ai_phrases_warning(self) -> str:
        """
        Generate a warning about avoiding common AI phrases.
        
        Returns:
            String with AI phrases to avoid during humanization
        """
        # Take a sample of phrases to include in the prompt (to keep prompt size manageable)
        sample_phrases = self.common_ai_phrases_to_avoid[:15]  # All phrases
        phrases_text = '", "'.join(sample_phrases)
        
        return f"""
IMPORTANT - Avoid Overused AI Phrases: Try to avoid or minimize the use of common AI-generated phrases unless absolutely necessary for the meaning. These include phrases like: "{phrases_text}", and similar corporate/AI-sounding language. Use more natural, conversational alternatives when possible."""
    
    def _humanize_paragraph(self, paragraph: str, iteration: int = 1) -> str:
        """
        Humanize a single paragraph using Groq API.
        
        Args:
            paragraph: Paragraph text to humanize
            iteration: Current iteration number (affects the prompt strategy)
            
        Returns:
            Humanized paragraph text
        """
        # Get AI phrases warning
        ai_phrases_warning = self._get_ai_phrases_warning()
        
        # Different prompts for different iterations
        if iteration == 1:
            system_prompt = f"""You are an expert text humanizer. Your task is to rewrite AI-generated text to make it sound more natural, conversational, and human-like.

Key requirements:
1. Make the text easily readable (aim for Flesch Reading Ease score of 80+)
2. Use simple, clear language that flows naturally
3. Avoid robotic or overly formal language
4. Replace formal transitions with natural ones (swap "Furthermore" → "Plus", "Moreover" → "And", "In conclusion" → "So")
5. Vary sentence length and structure within the same overall length
6. Make it sound like a knowledgeable human wrote it
7. Preserve all important information and meaning
8. CRITICAL: Maintain the same word count (±15% tolerance). Do not add extra content, examples, or explanations - only rewrite existing content.
{ai_phrases_warning}

IMPORTANT: Only return the humanized text. Do not include any explanations, introductions, or phrases like "Here is the humanized text". Just provide the rewritten content directly.

Focus on making the text accessible and engaging while keeping the same length and information density."""

        elif iteration == 2:
            system_prompt = f"""You are a skilled editor focused on making text more human and relatable. Refine the text further by:

1. Injecting personality and warmth through word choice, not additional words
2. Using more conversational connectors ("And", "But", "So", "Plus") to replace formal ones
3. Simplifying complex sentences by breaking them down or using simpler words
4. Converting passive voice to active voice where possible
5. Making the tone more approachable and less academic through word substitution
6. CRITICAL: Maintain the same word count (±15% tolerance). Focus on replacing words and restructuring sentences, not adding content.
{ai_phrases_warning}

IMPORTANT: Only return the refined text. Do not include any explanations, introductions, or phrases like "Here is the refined text". Just provide the improved content directly.

Keep the core message intact while making it sound like natural human communication within the same length constraints."""

        else:  # iteration 3+
            system_prompt = f"""You are a final polish editor. Make the text sound completely natural and human by:

1. Ensuring perfect flow between sentences through better transitions
2. Using everyday language instead of formal terms (replace complex words with simpler equivalents)
3. Making sure it sounds like spoken conversation when read aloud
4. Removing any remaining artificial-sounding phrases
5. Ensuring the text feels warm and engaging through word choice
6. CRITICAL: Maintain the same word count (±15% tolerance). This is a polishing pass - refine existing content without expansion.
{ai_phrases_warning}

IMPORTANT: Only return the final polished text. Do not include any explanations, introductions, or phrases like "Here is the final version". Just provide the polished content directly.

This is the final pass - make it sound like a friendly, knowledgeable person explaining something, but keep it the same length."""

        original_word_count = self._count_words(paragraph)
        user_prompt = f"Please humanize this text according to the guidelines above. Original word count: {original_word_count} words. Target word count: {original_word_count} words (±15% tolerance = {int(original_word_count * 0.85)}-{int(original_word_count * 1.15)} words).\n\n{paragraph}"
        
        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=2000
            )
            
            result = completion.choices[0].message.content.strip()
            # Sanitize the result to prevent Unicode encoding errors
            return sanitize_unicode_text(result)
        
        except Exception as e:
            # Log error and return original paragraph
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}]   \u26a0\ufe0f  API Error in iteration {iteration}: {str(e)}")
            print(f"[{timestamp}]   \ud83d\udd04 Falling back to original text for this iteration")
            # Also sanitize the fallback text
            return sanitize_unicode_text(paragraph)
    
    def _final_coherence_pass(self, text: str) -> str:
        """
        Perform a final coherence pass on the entire humanized text to improve flow and unity.
        
        Args:
            text: The fully humanized text from paragraph-wise processing
            
        Returns:IMAGE_DATA_URL
            Final coherent and humanized text
        """
        # Get AI phrases warning
        ai_phrases_warning = self._get_ai_phrases_warning()
        
        # Create comprehensive final pass prompt
        system_prompt = f"""You are an expert text editor specializing in making AI-generated content sound completely natural and human. Your task is to perform a final coherence pass on text that has already been humanized paragraph by paragraph.

CRITICAL OBJECTIVES:
1. **Maintain ALL Content**: Preserve every piece of information, data, examples, and key points from the original text. Do not remove, summarize, or omit anything important.

2. **Improve Flow & Coherence**: Ensure smooth transitions between paragraphs and ideas while maintaining the existing structure and information density.

3. **Enhance Natural Language**: Make the text sound like it was written by a knowledgeable human having a natural conversation with the reader.

4. **Word Count Preservation**: Maintain approximately the same word count (±10% tolerance). This is content refinement, not expansion or reduction.

SPECIFIC TASKS:
• Fix any awkward transitions between paragraphs that resulted from separate processing
• Ensure consistent tone and voice throughout the entire piece  
• Smooth out any repetitive phrasing that may have occurred across different sections
• Make sure the text flows as one cohesive piece rather than separate paragraphs
• Replace any remaining formal or robotic language with natural alternatives
• Ensure the writing feels warm, engaging, and conversational throughout

{ai_phrases_warning}

FORMATTING RULES:
• Maintain the original paragraph structure and spacing
• Keep any existing formatting, lists, or organizational elements
• Preserve technical terms and specific information exactly as provided
• Only return the refined text - no explanations, introductions, or meta-commentary

Remember: This is a COHERENCE and FLOW improvement pass, not a content change pass. Keep everything substantial while making it read like natural human writing."""

        original_word_count = self._count_words(text)
        user_prompt = f"Please perform a final coherence pass on this text to make it flow naturally as one unified piece while preserving all content and information. Original word count: {original_word_count} words. Target: maintain same length.\n\n{text}"
        
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] 🎯 FINAL COHERENCE PASS")
            print(f"[{timestamp}]   📝 Processing entire text ({original_word_count} words)")
            print(f"[{timestamp}]   🎯 Goal: Improve flow and unity while preserving content")
            
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.6,  # Slightly lower temperature for more consistent results
                max_tokens=4000
            )
            
            result = completion.choices[0].message.content.strip()
            # Sanitize the result to prevent Unicode encoding errors
            result = sanitize_unicode_text(result)
            final_word_count = self._count_words(result)
            
            print(f"[{timestamp}]   ✅ Final coherence pass completed")
            print(f"[{timestamp}]   📊 Word count: {original_word_count} → {final_word_count} ({final_word_count - original_word_count:+d})")
            
            return result
            
        except Exception as e:
            # Log error and return original text
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}]   ⚠️  API Error in final coherence pass: {str(e)}")
            print(f"[{timestamp}]   🔄 Returning text without final pass")
            # Sanitize the fallback text as well
            return sanitize_unicode_text(text)
    
    def humanize_text(self, text: str, n_iterations: int = 2) -> str:
        """
        Humanize the entire text by processing paragraphs with multiple iterations.
        
        Args:
            text: Input text to humanize
            n_iterations: Number of humanization passes (1-5 recommended)
            
        Returns:
            Fully humanized text
        """
        if not text.strip():
            return text
        
        # Sanitize input text to prevent Unicode encoding errors
        text = sanitize_unicode_text(text)
        
        # Validate n_iterations
        n_iterations = max(1, min(n_iterations, 5))  # Limit between 1-5
        
        # Detect paragraphs
        paragraphs = self._detect_paragraphs(text)
        
        if not paragraphs:
            return text
        
        # Print initial status
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] 🚀 HUMANIZATION STARTED")
        print(f"[{timestamp}] 📝 Processing {len(paragraphs)} chunk{'s' if len(paragraphs) != 1 else ''} with {n_iterations} iteration{'s' if n_iterations != 1 else ''}")
        print(f"[{timestamp}] 🤖 Using model: {self.model}")
        print(f"[{timestamp}] " + "="*50)
        
        # Process each paragraph through multiple iterations
        humanized_paragraphs = []
        
        for i, paragraph in enumerate(paragraphs):
            current_paragraph = paragraph
            chunk_num = i + 1
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📄 Processing chunk {chunk_num}/{len(paragraphs)} ({len(paragraph.split())} words)")
            
            # Apply multiple iterations of humanization
            for iteration in range(1, n_iterations + 1):
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   🔄 Iteration {iteration}/{n_iterations} for chunk {chunk_num}")
                sys.stdout.flush()  # Ensure immediate output
                
                current_paragraph = self._humanize_paragraph(current_paragraph, iteration)
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   ✅ Completed iteration {iteration}/{n_iterations} for chunk {chunk_num}")
            
            humanized_paragraphs.append(current_paragraph)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✨ Finished processing chunk {chunk_num}/{len(paragraphs)}")
        
        # Join paragraphs back together with double newlines
        result = '\n\n'.join(humanized_paragraphs)
        
        # Skip final coherence pass - preserve paragraph-wise humanization results
        # Final LLM pass removed as requested - paragraph-wise approach is final
        # if self.last_pass:
        #     print(f"\n[{datetime.now().strftime('%H:%M:%S')}] " + "="*50)
        #     print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 STARTING FINAL COHERENCE PASS")
        #     print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Goal: Improve flow and unity across entire text")
        #     
        #     result = self._final_coherence_pass(result)
        #     
        #     print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Final coherence pass completed")
        
        # Print completion status
        final_timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{final_timestamp}] " + "="*50)
        print(f"[{final_timestamp}] 🎉 HUMANIZATION COMPLETED SUCCESSFULLY")
        print(f"[{final_timestamp}] 📊 Processed {len(paragraphs)} chunks with {n_iterations} iterations each")
        
        # Calculate total operations (no final pass)
        total_operations = len(paragraphs) * n_iterations
        print(f"[{final_timestamp}] 📝 Total operations: {total_operations} API calls (paragraph-wise humanization only)")
            
        print(f"[{final_timestamp}] 💾 Output ready for delivery\n")
        
        return result
    
    def quick_humanize(self, text: str) -> str:
        """
        Quick single-pass humanization for faster processing.
        
        Args:
            text: Input text to humanize
            
        Returns:
            Humanized text
        """
        return self.humanize_text(text, n_iterations=1)


# Example usage and testing
if __name__ == "__main__":
    # Test the humanizer
    sample_text = """
    Artificial intelligence represents a transformative technological advancement that has the potential to revolutionize numerous industries. Furthermore, machine learning algorithms enable systems to learn from data and improve their performance over time. Moreover, the implementation of AI solutions can significantly enhance operational efficiency and decision-making processes.

    In conclusion, organizations that strategically integrate artificial intelligence into their workflows will likely gain competitive advantages. Additionally, the continuous evolution of AI technologies suggests that future applications will be even more sophisticated and capable.
    """
    
    try:
        # Initialize the humanizer
        humanizer = HumanizeTextWithGroq()
        
        print("Original text:")
        print(sample_text)
        original_word_count = humanizer._count_words(sample_text)
        print(f"Original word count: {original_word_count}")
        print("\n" + "="*50 + "\n")
        
        # Humanize with 2 iterations
        humanized = humanizer.humanize_text(sample_text, n_iterations=2)
        
        print("Humanized text:")
        print(humanized)
        humanized_word_count = humanizer._count_words(humanized)
        print(f"\nHumanized word count: {humanized_word_count}")
        
        # Calculate word count change
        change_percent = ((humanized_word_count - original_word_count) / original_word_count) * 100
        print(f"Word count change: {change_percent:+.1f}%")
        
        if abs(change_percent) <= 15:
            print("✓ Word count preserved within ±15% tolerance")
        else:
            print("⚠ Word count exceeded ±15% tolerance")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have GROQ_API_KEY set in your environment variables.")