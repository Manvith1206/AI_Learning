# Enhanced RAG Evaluation Prompts with Advanced Prompt Engineering

FAITHFULNESS_CALCULATION_PROMPT_TEMPLATE = """You are an expert fact-checker analyzing the relationship between context and statements.

**Task**: Determine if the given statement is directly supported by the provided context.

**Instructions**:
- Only respond "yes" if the statement can be directly inferred from the context
- Respond "no" if the statement contradicts the context or requires external knowledge
- Ignore minor stylistic differences; focus on semantic meaning
- The statement must be factually grounded in the context, not just topically related

**Context**:
{context}

**Statement to Evaluate**:
{statement}

**Your Response** (only "yes" or "no"):"""

CONTEXT_PRECISION_CALCULATION_PROMPT_TEMPLATE = """You are an expert information retrieval analyst evaluating context relevance.

**Task**: Determine if the provided context chunk contains information relevant to answering the given question.

**Evaluation Criteria**:
- Context is relevant if it contains facts, data, or information that directly helps answer the question
- Context is relevant if it provides necessary background or supporting details
- Context is NOT relevant if it only shares topical similarity without providing useful information
- Consider partial relevance as relevant ("yes")

**Question**:
{question}

**Context Chunk to Evaluate**:
{context_chunk}

**Assessment** (only "yes" or "no"):"""

CONTEXT_RECALL_CALCULATION_PROMPT_TEMPLATE = """You are an expert evaluator assessing information completeness.

**Task**: Determine if the ground truth statement can be logically inferred from the provided context.

**Evaluation Guidelines**:
- Respond "yes" if the context contains sufficient information to derive the statement
- Respond "no" if critical information is missing from the context
- Consider reasonable inferences based on the provided information
- The context doesn't need to contain the exact wording, but must support the statement's claims

**Context**:
{context}

**Ground Truth Statement**:
{statement}

**Can this statement be inferred from the context?** (only "yes" or "no"):"""

ANSWER_RELEVANCY_CALCULATION_PROMPT_TEMPLATE = """You are an expert question generation specialist creating evaluation questions.

**Task**: Generate {num_questions} questions that are semantically similar to the original question and would be answered by the same response.

**Requirements for Generated Questions**:
1. **Semantic Equivalence**: Questions must have the same core meaning and intent
2. **Answer Compatibility**: All questions should be answerable by the provided answer
3. **Lexical Variation**: Use different wording while preserving meaning
4. **Question Type Consistency**: Maintain the same question format (what/how/why/when, etc.)
5. **Entity Preservation**: Keep key entities and concepts from the original question

**Original Question**: 
{original_question}

**Answer**: 
{answer}

**Generated Questions**:
1. 
2. 
3. 
{additional_questions}

Note: Generate exactly {num_questions} questions, each on a new numbered line."""

# Alternative template for more structured output
ANSWER_RELEVANCY_STRUCTURED_TEMPLATE = """<role>Expert Question Generation Specialist</role>

<task>
Generate {num_questions} semantically equivalent questions that would receive the same answer as the original question.
</task>

<criteria>
- Preserve core meaning and intent
- Maintain question type (what/how/why/when)
- Use varied vocabulary and phrasing
- Ensure answer compatibility
- Keep essential entities and concepts
</criteria>

<input>
Original Question: {original_question}
Reference Answer: {answer}
</input>

<output_format>
1. [Question 1]
2. [Question 2]
3. [Question 3]
[Continue for {num_questions} total questions]
</output_format>

Generated Questions:"""

# Utility template for batch processing
BATCH_EVALUATION_TEMPLATE = """You are processing multiple evaluation tasks. For each item, provide only the requested response format.

**Task Type**: {task_type}
**Response Format**: {response_format}

{batch_items}

Responses:"""