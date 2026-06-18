import os
import sys
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

from buffer_client import BufferClient
from llm_client import LLMClient

# Load environment variables from .env if present (useful for local development)
load_dotenv()

BRAND_VOICE_FILE = "brand_voice.txt"
HISTORY_FILE = "history.json"

def get_env_var(name: str, required: bool = True) -> str:
    val = os.getenv(name)
    if required and not val:
        print(f"Error: Environment variable '{name}' is not set.", file=sys.stderr)
        sys.exit(1)
    return val or ""

def load_text_file(filepath: str) -> str:
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def load_history(filepath: str) -> list:
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: '{filepath}' contains invalid JSON. Starting with empty history.", file=sys.stderr)
        return []

def save_history(filepath: str, history: list):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def cmd_list_channels():
    api_key = get_env_var("BUFFER_API_KEY")
    client = BufferClient(api_key)
    
    print("Fetching organizations...")
    try:
        orgs = client.get_organizations()
    except Exception as e:
        print(f"Failed to fetch organizations: {e}", file=sys.stderr)
        sys.exit(1)
        
    if not orgs:
        print("No organizations found on this Buffer account.")
        return
        
    for org in orgs:
        org_id = org.get("id")
        org_name = org.get("name")
        print(f"\nOrganization: {org_name} (ID: {org_id})")
        print("-" * 50)
        
        try:
            channels = client.get_channels(org_id)
        except Exception as e:
            print(f"  Failed to fetch channels: {e}", file=sys.stderr)
            continue
            
        if not channels:
            print("  No social channels connected to this organization.")
            continue
            
        for ch in channels:
            print(f"  Channel Name: {ch.get('displayName')} ({ch.get('name')})")
            print(f"  Service     : {ch.get('service')}")
            print(f"  Channel ID  : {ch.get('id')}")
            print(f"  Queue Status: {'PAUSED' if ch.get('isQueuePaused') else 'ACTIVE'}")
            print("  " + "." * 30)

def run_generation_pipeline(mock: bool = False) -> tuple:
    """
    Executes the LLM pipeline: read history/brand voice -> generate -> score -> select -> polish.
    Returns: (polished_tweet, selected_idea)
    """
    base_url = None
    if mock:
        api_key = "mock"
        model_name = "mock-model"
    else:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            api_key = gemini_key
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            print("Using Google Gemini API.")
        else:
            api_key = get_env_var("OPENAI_API_KEY")
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print("Using OpenAI API.")
    
    print("Loading brand voice and history...")
    brand_voice = load_text_file(BRAND_VOICE_FILE)
    history = load_history(HISTORY_FILE)
    
    print(f"Initializing LLM client (Model: {model_name})...")
    llm = LLMClient(api_key, model_name, base_url=base_url)
    
    print("\n--- STEP 1: Generating 5 Ideas ---")
    ideas = llm.generate_ideas(brand_voice, history)
    for idea in ideas:
        print(f"ID {idea.get('id')} [{idea.get('topic')}]: {idea.get('idea_text')}")
        
    print("\n--- STEPS 2 & 3: Scoring Ideas ---")
    scores = llm.score_ideas(ideas, brand_voice)
    for score in scores:
        idea_id = score.get("id")
        total = (score.get("brand_alignment_score", 0) + 
                 score.get("engagement_score", 0) + 
                 score.get("originality_score", 0)) / 3.0
        print(f"ID {idea_id}: Brand: {score.get('brand_alignment_score')}/10, "
              f"Engagement: {score.get('engagement_score')}/10, "
              f"Originality: {score.get('originality_score')}/10 (Avg: {total:.2f})")
        print(f"  Reason: {score.get('reasoning')}")
        
    print("\n--- STEP 4: Selecting Best Idea ---")
    best_idea = llm.select_best_idea(ideas, scores)
    print(f"Selected ID {best_idea.get('id')} (Topic: {best_idea.get('topic')}) "
          f"with score {best_idea.get('composite_score'):.2f}")
    
    print("\n--- STEP 5: Rewriting & Polishing ---")
    polished_tweet = llm.polish_idea(best_idea, brand_voice)
    print("Polished Tweet Content:")
    print("-" * 40)
    print(polished_tweet)
    print("-" * 40)
    print(f"Length: {len(polished_tweet)} characters")
    
    return polished_tweet, best_idea

def cmd_test_generate(mock: bool = False):
    print("Running dry-run content generation pipeline...")
    run_generation_pipeline(mock)
    print("\nDry-run complete. No posts were scheduled.")

def cmd_run(mock: bool = False):
    print("Starting automated posting sequence...")
    # Generate post content
    polished_tweet, idea_details = run_generation_pipeline(mock)
    
    # Verify length constraints (strictly under 280 characters)
    if len(polished_tweet) > 280:
        print(f"Error: Generated tweet is too long ({len(polished_tweet)} chars). Aborting.", file=sys.stderr)
        sys.exit(1)
        
    if mock:
        buffer_key = "mock"
        channel_id = "mock"
        print("\n[MOCK] Bypassing Buffer API. Simulating queue additions.")
    else:
        buffer_key = get_env_var("BUFFER_API_KEY")
        channel_id = get_env_var("BUFFER_CHANNEL_ID")
    
    print(f"\nScheduling post on Buffer (Channel ID: {channel_id})...")
    
    try:
        if mock:
            post_info = {"id": "mock_buffer_post_12345"}
        else:
            buffer = BufferClient(buffer_key)
            post_info = buffer.create_post(channel_id, polished_tweet)
        print(f"Success! Post queued. Buffer Post ID: {post_info.get('id')}")
    except Exception as e:
        print(f"Failed to schedule post on Buffer: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("Updating history file...")
    history = load_history(HISTORY_FILE)
    history.append({
        "text": polished_tweet,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "topic": idea_details.get("topic"),
        "score": idea_details.get("composite_score")
    })
    save_history(HISTORY_FILE, history)
    print(f"History updated. Total posts stored: {len(history)}")

def main():
    parser = argparse.ArgumentParser(description="Automated Buffer Twitter/X Posting Bot")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list-channels", action="store_true", help="List connected Buffer profiles and organization details")
    group.add_argument("--test-generate", action="store_true", help="Simulate tweet generation without scheduling on Buffer")
    group.add_argument("--run", action="store_true", help="Generate post and queue to Buffer")
    
    parser.add_argument("--mock", action="store_true", help="Use mock responses for OpenAI and Buffer to run a dry test without API keys")
    
    args = parser.parse_args()
    
    if args.list_channels:
        cmd_list_channels()
    elif args.test_generate:
        cmd_test_generate(args.mock)
    elif args.run:
        cmd_run(args.mock)

if __name__ == "__main__":
    main()
