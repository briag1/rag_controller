def compute_reward_context_relevancy(retrieved_docs: list[NodeWithScore], ground_truth_answers):
    """
    Computes reward based on retrieved documents' overlap with ground truth answers.
    """
    retrieved_texts = [doc.node.text for doc in retrieved_docs]
    reward = sum(any(gt in doc for gt in ground_truth_answers) for doc in retrieved_texts) / len(retrieved_docs)
    return reward