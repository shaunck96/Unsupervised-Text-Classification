import ast
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
import pandas as pd
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
MODEL_ID = "microsoft/Phi-3-small-8k-instruct"
MAX_TOKENS_PER_BATCH = 3000  
MAX_NEW_TOKENS = 100  

def load_transcription(file_path: str) -> pd.DataFrame:
    """Load transcription data from a parquet file."""
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Error loading transcription data: {e}")
        raise

def merger(anonymized_list: List[Dict]) -> str:
    """Merge anonymized transcription segments."""
    return " ".join(trans['text'] for trans in anonymized_list)

def conversation_context_parser(conversation_list: List[str], token_count_list: List[int]) -> List[str]:
    """Parse conversation contexts into batches based on token count."""
    conversation_context_list = []
    conversation_context = ""
    total_tokens = 0
    
    for conversation, token_count in zip(conversation_list, token_count_list):
        conversation = f"Conversation: {conversation}"
        
        if total_tokens + token_count > MAX_TOKENS_PER_BATCH:
            conversation_context_list.append(conversation_context.strip())
            conversation_context = ""
            total_tokens = 0
        
        conversation_context += f"{conversation}\n"
        total_tokens += token_count

    if conversation_context:
        conversation_context_list.append(conversation_context.strip())

    return conversation_context_list

def initialize_model_and_tokenizer(model_id: str):
    """Initialize the model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return tokenizer, pipe
    except Exception as e:
        logger.error(f"Error initializing model and tokenizer: {e}")
        raise

def extract_topics(conversation_context: str, pipe):
    """Extract topics from a conversation context."""
    try:
        system_prompt = f"""
        You are an advanced AI assistant tasked with analyzing customer service conversations from an electrical utilities company. Your goal is to extract key topics. Follow these structured steps:

        1. **Read and Understand**: Carefully read each conversation to fully grasp the context and flow of the dialogue.
        2. **Topic Identification**: For each conversation, identify the main topics discussed (focus on issues like billing, outages, service requests, etc.). The topics should be specific and relevant to electrical utilities.
        3. **Ensure Topic Specificity**: Avoid generic categories. Use concrete, concise phrases that precisely capture the core issue of the conversation (e.g., 'Billing confusion', 'Storm outage', 'Service upgrade request').
        4. **Check for Redundancies**: Ensure that each topic is unique and does not repeat across conversations unless it explicitly applies to multiple conversations.
        5. **Topic Limit**: Each conversation should result in 3-5 topics. If fewer are applicable, list at least one topic.
        6. **Final Output Format**: Return the topics in the format of a Python list of strings. Each string represents one topic.

        Conversation to analyze:
        {conversation_context}

        Output only the Python list of topics. Example:
        ['Billing confusion', 'Storm outage', 'Account update request']
        """

        messages = [{"role": "user", "content": system_prompt}]
        generation_args = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        output = pipe(messages, **generation_args)
        output_text = output[0]['generated_text']

        try:
            topics = ast.literal_eval(output_text)
            return topics if isinstance(topics, list) else []
        except Exception:
            logger.warning(f"Failed to parse topics: {output_text}")
            return []
    except Exception as e:
        logger.error(f"Error in extract_topics: {e}")
        return []

def generate_master_dict(all_topics: List[List[str]], pipe):
    """Generate master dictionary using a prompting strategy."""
    try:
        # Flatten the list of topics
        all_topics_flat = [topic for sublist in all_topics for topic in sublist]

        system_prompt = f"""
        You are an AI tasked with organizing topics from customer service conversations for an electrical utilities company. Follow these structured steps:

        1. **Category Creation**: First, think about the common issues faced by customers in the electrical utilities industry. Create 3-5 high-level categories that broadly cover these topics. The categories should be clear, concise, and relevant (e.g., 'Billing', 'Outages', 'Service Requests', 'Account Management').
        2. **Assign Topics to Categories**: For each topic, carefully determine which category it belongs to. The assignment should make logical sense and be as specific as possible. Ensure no topic is placed under multiple categories.
        3. **Check for Duplicates**: Ensure no topic is repeated within or across categories.
        4. **Final Format**: Return the categorized topics in a Python dictionary. Each key will be a category (as a string), and the value will be a list of associated topics.

        Topics to categorize:
        {all_topics_flat}

        Example output format:
        {{
            'Billing': ['Late fees', 'Payment methods'],
            'Outages': ['Power restoration', 'Storm-related outages'],
            'Service Requests': ['New connection', 'Meter reading']
        }}

        Output only the Python dictionary, without any additional text.
        """

        messages = [{"role": "user", "content": system_prompt}]
        generation_args = {
            "max_new_tokens": MAX_NEW_TOKENS * 2,  # Increase token limit for this task
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        output = pipe(messages, **generation_args)
        output_text = output[0]['generated_text']

        try:
            master_dict = ast.literal_eval(output_text)
            return master_dict if isinstance(master_dict, dict) else {}
        except Exception:
            logger.warning(f"Failed to parse master dictionary: {output_text}")
            return {}
    except Exception as e:
        logger.error(f"Error in generate_master_dict: {e}")
        return {}

def process_conversations(conversation_context_list: List[str], pipe) -> Dict[str, List[str]]:
    """Process conversations and generate master topic dictionary."""
    try:
        all_topics = []
        for i, conversation in enumerate(conversation_context_list):
            topics = extract_topics(conversation, pipe)
            all_topics.append(topics)
            print(f"Conversation {i+1} Topics:")
            print(topics)
            print()

        # Generate master dictionary using prompting strategy
        master_dict = generate_master_dict(all_topics, pipe)

        return master_dict
    except Exception as e:
        logger.error(f"Error in process_conversations: {e}")
        raise

def main():
    try:
        # Load transcription data
        ip_path = r"/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/Production_Code/transcriptions/gpu_transcriptions_redacted_call_sid_based.parquet"
        transcription = load_transcription(ip_path).iloc[:15, :]
        print("Transcription data loaded successfully.")

        # Initialize model and tokenizer
        tokenizer, pipe = initialize_model_and_tokenizer(MODEL_ID)
        print("Model and tokenizer initialized successfully.")

        # Prepare conversation data
        transcription['transcription'] = transcription['anonymized'].apply(merger)
        transcription['token_count'] = transcription['transcription'].apply(lambda x: len(tokenizer.encode(x)))

        conversation_list = list(transcription['transcription'])
        token_count_list = list(transcription['token_count'])
        conversation_context_list = conversation_context_parser(conversation_list, token_count_list)
        print(f"Prepared {len(conversation_context_list)} conversation contexts.")

        # Process conversations
        final_master_topic_dict = process_conversations(conversation_context_list, pipe)

        # Display the final master topic dictionary
        print("\nFinal Master Topic Dictionary:")
        for category, subtopics in final_master_topic_dict.items():
            print(f"\n{category}:")
            for subtopic in subtopics:
                print(f"  - {subtopic}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
