# refer to https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html

api_request_list = {
    'us.amazon.nova-pro-v1:0': {
        "modelId": "us.amazon.nova-pro-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "schemaVersion": "messages-v1",
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": ""}]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 4096,
                "temperature": 0,
                "topP": 1,
                "topK": 100
            }
        }
    },
    'us.anthropic.claude-3-5-sonnet-20241022-v2:0': {
        "modelId": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": ""
                        }
                    ]
                }
            ],
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "temperature": 0.5,
            "top_p": 1,
            "top_k": 250,
            "stop_sequences": ["\n\nHuman:"]
        }
    },
    'us.anthropic.claude-3-5-haiku-20241022-v1:0': {
        "modelId": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": ""
                        }
                    ]
                }
            ],
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "temperature": 0.5,
            "top_p": 1,
            "top_k": 250,
            "stop_sequences": ["\n\nHuman:"]
        }
    },
}


model_ids = list(api_request_list.keys())
