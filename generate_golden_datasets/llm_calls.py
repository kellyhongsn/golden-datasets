from typing import Any

def generate_query(client: Any, corpus: str) -> str:

    SYSTEM_INSTRUCTION = "You are an assistant specialized in generating queries to curate a high-quality synthetic dataset"
    
    PROMPT = f"""
        Based on the following piece of information:
        <corpus>
        {corpus}
        <corpus>

        Please generate a query relevant to the information provided above.

        Simply output the query without any additional words in this format:
        <format>
        [query]
        <format>
        """
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        temperature=1,
        system=SYSTEM_INSTRUCTION,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    }
                ]
            }
        ]
    )

    return message.content[0].text


def generate_query_with_example(client: Any, query: str, corpus: str) -> str:

    SYSTEM_INSTRUCTION = "You are an assistant specialized in generating queries to curate a high-quality synthetic dataset"

    PROMPT = f"""
    Based on the following contextual information:
    <corpus>
    {corpus}
    <corpus>

    This would be an example query that would be good for this kind of context:
    <query>
    {query}
    <query>

    Please generate one additional query that is distinct from the example, but is still relevant to the corpus. This point is very important, ensure that the generated query does not repeat the given example query.

    Simply output the query without any additional words in this format:
    <format>
    [query]
    <format>
    """

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        temperature=1,
        system=SYSTEM_INSTRUCTION,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    }
                ]
            }
        ]
    )

    return message.content[0].text