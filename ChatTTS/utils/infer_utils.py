
import torch
import torch.nn.functional as F

    
class CustomRepetitionPenaltyLogitsProcessorRepeat():

    def __init__(self, penalty: float, max_input_ids, past_window):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        input_ids = input_ids[:, -self.past_window:]
        freq = F.one_hot(input_ids, scores.size(1)).sum(1)
        freq[self.max_input_ids:] = 0
        alpha = self.penalty**freq
        scores = torch.where(scores < 0, scores*alpha, scores/alpha)

        return scores
    
class CustomRepetitionPenaltyLogitsProcessor():

    def __init__(self, penalty: float, max_input_ids, past_window):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        input_ids = input_ids[:, -self.past_window:]
        score = torch.gather(scores, 1, input_ids)
        _score = score.detach().clone()
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        score[input_ids>=self.max_input_ids] = _score[input_ids>=self.max_input_ids]
        scores.scatter_(1, input_ids, score)
        
        return scores