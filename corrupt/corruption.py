import numpy as np
# import skvideo as skv
from abc import ABC, abstractmethod



class Corruption():
    def __init__(self) -> None:    
        pass

    @staticmethod
    def validate(self, video_path, open_handle=None):
        #TODO check if video can still be opened normally
        try:
            video=open_handle(video_path)
        except:
            return False
        return True

    
    @staticmethod
    def visualize(self, file_path: str):
        #TODO visualize video
        pass

    @abstractmethod
    def corrupt(self, file):
        #TODO corrupt video
        pass

    def corrupt_batch(self, file_path: list, num_workers: int):
        #TODO corrupt batch of videos
        pass


