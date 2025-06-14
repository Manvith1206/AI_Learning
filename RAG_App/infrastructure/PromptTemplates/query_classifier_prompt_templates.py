BASE_QUERY_CLASSIFER_PROMPT = """
# ROLE
You are an AI assistant that classifies user queries into one of three categories:
    1. greeting - if the user is simply greeting or making small talk
    2. relevant - if the query is relevant to the provided context
    3. irrelevant - if the query is unrelated to the provided context
    
        User query: {query_text}
"""

QUERY_CLASSIFIER_WITH_CONTEXTS_PROMPT = """
Context from documents:
            {context_text}
            
            Based on the user query and the context provided, classify the query as 'greeting', 'relevant', or 'irrelevant'.
            If the query is asking for information that might be in the context, classify as 'relevant'.
            If the query is completely unrelated to the context, classify as 'irrelevant'.
            If the query is just a greeting or small talk, classify as 'greeting'.
            """


QUERY_CLASSIFIER_WITHOUT_CONTEXTS_PROMPT = """
            No context is available. Determine if the query is a greeting or small talk.
            If it's a greeting or small talk, classify as 'greeting'.
            Otherwise, classify as 'relevant' assuming the user is asking a substantive question.
"""
