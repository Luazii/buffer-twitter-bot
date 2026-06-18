import json
from openai import OpenAI
from typing import Dict, List, Any, Tuple

class LLMClient:
    """
    Handles LLM content generation steps using the OpenAI API.
    Implements a multi-step workflow: Generate -> Score -> Select -> Polish.
    """
    def __init__(self, api_key: str, model_name: str, base_url: str = None):
        if not api_key:
            raise ValueError("API Key is required.")
        self.api_key = api_key
        if api_key == "mock":
            self.client = None
        else:
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def _call_llm_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Helper method to call OpenAI with JSON response format.
        """
        if self.client is None:
            raise ValueError("LLMClient is in mock mode and cannot call live APIs.")
            
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Received empty response from OpenAI.")
        return json.loads(content)

    def generate_ideas(self, brand_voice: str, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 1: Generate 5 diverse post ideas based on brand voice and history.
        """
        if self.api_key == "mock":
            return [
                {"id": 1, "topic": "Developer Productivity", "idea_text": "Stop multitasking. Focus on one task, get it done, then move on. Your brain isn't a GPU."},
                {"id": 2, "topic": "AI Tools", "idea_text": "AI won't replace developers, but developers who use AI will replace those who don't. Learn to prompt and review code."},
                {"id": 3, "topic": "Remote Work", "idea_text": "Remote work is about trust, not tracking keystrokes. Output over presence, always."},
                {"id": 4, "topic": "Coding Practice", "idea_text": "Write clean code. Not because it makes you feel smart, but because the developer who maintains it in 6 months might be you."},
                {"id": 5, "topic": "Indie Hacking", "idea_text": "Start small. Launch early. Listen to feedback. Iterate. That's the secret to building products people actually want."}
            ]

        system_prompt = (
            "You are a professional content strategist. Your task is to generate 5 distinct, "
            "diverse, and compelling tweet ideas or drafts based on the provided brand voice "
            "guidelines and avoiding repetition of recent posts.\n\n"
            "Return a JSON object with the key 'ideas', containing a list of objects. "
            "Each object must have 'id' (1-5), 'topic' (brief description of the topic), "
            "and 'idea_text' (the draft tweet text, around 150-250 characters)."
        )

        history_str = "No recent posts."
        if history:
            # Format history to show the last 20 posts to avoid redundancy
            recent_posts = [h.get("text", "") for h in history[-20:] if h.get("text")]
            history_str = "\n".join([f"- {post}" for post in recent_posts])

        user_prompt = (
            f"--- BRAND VOICE GUIDELINES ---\n{brand_voice}\n\n"
            f"--- RECENT POSTS HISTORY (DO NOT REPEAT THESE) ---\n{history_str}\n\n"
            "Generate 5 unique ideas/drafts following the brand voice guidelines. "
            "Ensure the topics are diverse, the formats vary (e.g. some insights, some questions, "
            "some productivity tips, some code examples), and they do not repeat past ideas. "
            "Return the output in the specified JSON format."
        )

        data = self._call_llm_json(system_prompt, user_prompt)
        ideas = data.get("ideas")
        if not ideas or not isinstance(ideas, list) or len(ideas) == 0:
            raise Exception("Invalid ideas format returned from LLM.")
        return ideas

    def score_ideas(self, ideas: List[Dict[str, Any]], brand_voice: str) -> List[Dict[str, Any]]:
        """
        Step 2 & 3: Score the 5 ideas based on quality criteria.
        """
        if self.api_key == "mock":
            return [
                {"id": 1, "brand_alignment_score": 9, "engagement_score": 8, "originality_score": 7, "reasoning": "Classic productivity tip, matches brand voice perfectly."},
                {"id": 2, "brand_alignment_score": 8, "engagement_score": 9, "originality_score": 6, "reasoning": "Very high engagement potential, slightly cliché but solid."},
                {"id": 3, "brand_alignment_score": 9, "engagement_score": 7, "originality_score": 8, "reasoning": "Great remote work insight, very clean."},
                {"id": 4, "brand_alignment_score": 10, "engagement_score": 9, "originality_score": 8, "reasoning": "Highly relatable developer wisdom, excellent brand alignment."},
                {"id": 5, "brand_alignment_score": 7, "engagement_score": 8, "originality_score": 8, "reasoning": "Good indie hacker topic, slightly off core developer focus."}
            ]

        system_prompt = (
            "You are an expert editor and copywriter. Evaluate the provided tweet ideas. "
            "For each idea, assign scores from 1 to 10 for the following criteria:\n"
            "1. brand_alignment: How well it matches the tone and rules in the guidelines.\n"
            "2. engagement_potential: How likely it is to get likes, retweets, or replies (strong hook, clarity).\n"
            "3. originality: How fresh and interesting the angle is.\n\n"
            "Return a JSON object with the key 'scores', containing a list of objects. "
            "Each object must have 'id' (corresponding to the idea's id), 'brand_alignment_score' (integer 1-10), "
            "'engagement_score' (integer 1-10), 'originality_score' (integer 1-10), "
            "and 'reasoning' (a brief explanation of the score)."
        )

        ideas_str = "\n".join([f"ID {i.get('id')}: Topic: {i.get('topic')}\nText: {i.get('idea_text')}\n" for i in ideas])

        user_prompt = (
            f"--- BRAND VOICE GUIDELINES ---\n{brand_voice}\n\n"
            f"--- IDEAS TO SCORE ---\n{ideas_str}\n"
            "Score each of the 5 ideas objectively. Return the result in the specified JSON format."
        )

        data = self._call_llm_json(system_prompt, user_prompt)
        scores = data.get("scores")
        if not scores or not isinstance(scores, list) or len(scores) == 0:
            raise Exception("Invalid scores format returned from LLM.")
        return scores

    def select_best_idea(self, ideas: List[Dict[str, Any]], scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Step 4: Rank the ideas and select the highest scoring one.
        """
        score_map = {s.get("id"): s for s in scores}
        
        best_idea = None
        highest_score = -1.0
        
        for idea in ideas:
            idea_id = idea.get("id")
            score_data = score_map.get(idea_id)
            if not score_data:
                continue
            
            # Calculate composite score (equal weight)
            composite_score = (
                score_data.get("brand_alignment_score", 0) +
                score_data.get("engagement_score", 0) +
                score_data.get("originality_score", 0)
            ) / 3.0
            
            idea["composite_score"] = composite_score
            idea["score_details"] = score_data
            
            if composite_score > highest_score:
                highest_score = composite_score
                best_idea = idea
                
        if not best_idea:
            # Fallback to the first idea
            best_idea = ideas[0]
            
        return best_idea

    def polish_idea(self, selected_idea: Dict[str, Any], brand_voice: str) -> str:
        """
        Step 5: Rewrite and polish the selected idea for high engagement, formatting, and character limits.
        """
        if self.api_key == "mock":
            text = selected_idea.get("idea_text", "")
            return f"{text}\n\nSimple as that. Agree?"

        system_prompt = (
            "You are a master tweet writer. Your job is to take a draft tweet concept and rewrite it "
            "into a polished, high-engagement version.\n\n"
            "Crucial Constraints:\n"
            "1. The final post MUST be strictly under 280 characters (usually aiming for 200-260 to be safe).\n"
            "2. Use line breaks to make it highly readable and skimmable.\n"
            "3. Ensure the hook (first line) is compelling and clicks with readers.\n"
            "4. Do NOT use emojis unless they are highly relevant, and never use more than one.\n"
            "5. NO HASHTAGS.\n"
            "6. Keep the tone natural and authentic.\n\n"
            "Return a JSON object with a single key 'polished_tweet' containing the final polished tweet text."
        )

        user_prompt = (
            f"--- BRAND VOICE GUIDELINES ---\n{brand_voice}\n\n"
            f"--- SELECTED DRAFT TWEET ---\nTopic: {selected_idea.get('topic')}\nText: {selected_idea.get('idea_text')}\n\n"
            "Rewrite and polish this tweet. Ensure it follows all constraints and is ready to post. "
            "Return the output in the specified JSON format."
        )

        data = self._call_llm_json(system_prompt, user_prompt)
        polished_tweet = data.get("polished_tweet")
        if not polished_tweet:
            raise Exception("Invalid polished tweet format returned from LLM.")
            
        # Verify length constraint
        polished_tweet = polished_tweet.strip()
        attempts = 0
        while len(polished_tweet) > 280 and attempts < 3:
            attempts += 1
            print(f"Warning: Polished tweet is too long ({len(polished_tweet)} characters). Asking LLM to shorten (Attempt {attempts}/3)...")
            shorten_prompt = (
                f"The following polished tweet is {len(polished_tweet)} characters, which violates the strict 280-character limit:\n\n"
                f"\"{polished_tweet}\"\n\n"
                f"Please rewrite and shorten this tweet so it is strictly under 280 characters. "
                f"Ensure you preserve the strong hook, line breaks, and insightful brand voice. "
                f"Do not use hashtags, and return the output in the same JSON format."
            )
            try:
                data = self._call_llm_json(system_prompt, shorten_prompt)
                polished_tweet = data.get("polished_tweet", "").strip()
            except Exception as e:
                print(f"Error during shortening attempt: {e}")
                break

        # Final safety truncation
        if len(polished_tweet) > 280:
            print("Warning: Tweet is still too long after 3 shortening attempts. Truncating to 277 characters + '...'")
            polished_tweet = polished_tweet[:277] + "..."
            
        return polished_tweet
