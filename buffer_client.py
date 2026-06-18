import requests
from typing import Dict, List, Any, Optional

class BufferClient:
    """
    A client wrapper for the Buffer GraphQL API (https://api.buffer.com).
    Uses personal access tokens for authentication.
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Buffer API Key is required.")
        self.api_key = api_key
        self.url = "https://api.buffer.com"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _execute_query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes a GraphQL query or mutation.
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = requests.post(self.url, headers=self.headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Check for top-level GraphQL errors
        if "errors" in result:
            errors = result["errors"]
            error_msgs = [err.get("message", "Unknown GraphQL error") for err in errors]
            raise Exception(f"Buffer GraphQL Error: {'; '.join(error_msgs)}")
            
        return result.get("data", {})

    def get_organizations(self) -> List[Dict[str, Any]]:
        """
        Retrieves the organizations associated with the authenticated user.
        """
        query = """
        query GetAccountOrganizations {
          account {
            organizations {
              id
              name
            }
          }
        }
        """
        data = self._execute_query(query)
        account = data.get("account") or {}
        return account.get("organizations") or []

    def get_channels(self, organization_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the social media channels connected under an organization.
        """
        query = f"""
        query GetChannels {{
          channels(input: {{ organizationId: "{organization_id}" }}) {{
            id
            name
            displayName
            service
            avatar
            isQueuePaused
          }}
        }}
        """
        data = self._execute_query(query)
        return data.get("channels") or []

    def create_post(self, channel_id: str, text: str) -> Dict[str, Any]:
        """
        Queues a text post (tweet) to a specific channel.
        Uses automatic scheduling (adds to the queue).
        """
        mutation = """
        mutation CreatePost($input: CreatePostInput!) {
          createPost(input: $input) {
            __typename
            ... on PostActionSuccess {
              post {
                id
                text
              }
            }
            ... on MutationError {
              message
            }
          }
        }
        """
        variables = {
            "input": {
                "channelId": channel_id,
                "text": text,
                "schedulingType": "automatic",
                "mode": "addToQueue"
            }
        }
        
        data = self._execute_query(mutation, variables)
        create_post_result = data.get("createPost") or {}
        typename = create_post_result.get("__typename")
        
        if typename == "MutationError":
            error_msg = create_post_result.get("message", "Unknown mutation error")
            raise Exception(f"Buffer Mutation Error: {error_msg}")
            
        if typename == "PostActionSuccess":
            return create_post_result.get("post") or {}
            
        raise Exception(f"Unexpected response type from createPost: {typename}")
