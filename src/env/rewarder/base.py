
from abc import ABC, abstractmethod
from typing import Any


class Rewarder(ABC):
  @abstractmethod
  def reward(self, *args: Any, **kwargs:Any) -> tuple[float, dict]:
    ...