from langchain.evaluation import load_evaluator

def reasonscore_evaluation(question, answer, llm):
    accuracy_criteria = {
        "accuracy": """
    Score 1: The answer is completely unrelated to the reference.
    Score 3: The answer has minor relevance but does not align with the reference.
    Score 5: The answer has moderate relevance but contains inaccuracies.
    Score 7: The answer aligns with the reference but has minor errors or omissions.
    Score 10: The answer is completely accurate and aligns perfectly with the reference."""
    }

    evaluator = load_evaluator(
        "labeled_score_string",
        criteria=accuracy_criteria,
        llm=llm,
    )

    # Correct
    eval_result = evaluator.evaluate_strings(
        prediction=answer,
        reference="",
        input=question,
    )
    return eval_result
