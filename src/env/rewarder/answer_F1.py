
class RewarderF1(Rewarder):

  def reward(self, answer:str, reference_answer:list[list[str]]) -> tuple[float, dict]:
      total_correct = 0
      answer_list = answer.split("/n")
      id_unmatched_references_answers = set(range(len(reference_answer)))
      reference_answer_copy = np.array(reference_answer, dtype=object)
      for answer in answer_list:
        for id_acceptable, acceptable_answers in enumerate(reference_answer_copy[list(id_unmatched_references_answers)]):
          if any(acceptable_answer in answer for acceptable_answer in acceptable_answers):
            total_correct += 1.0
            id_unmatched_references_answers.remove(list(id_unmatched_references_answers)[id_acceptable])
            break

      precision = total_correct/len(answer_list)
      recall = total_correct/len(reference_answer)
      if precision + recall == 0:
        return 0, {"precison": precision, "recall": recall}
      return 2 * (precision * recall) / (precision + recall), {"precison": precision, "recall": recall}