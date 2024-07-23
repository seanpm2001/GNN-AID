from defense.defense_base import Defender


class EvasionDefender(Defender):
    def __init__(self, **kwargs):
        super().__init__()

    def pre_epoch(self, **kwargs):
        pass

    def post_epoch(self, **kwargs):
        pass


class EmptyEvasionDefender(EvasionDefender):
    name = "EmptyEvasionDefender"

    def pre_epoch(self, **kwargs):
        pass

    def post_epoch(self, **kwargs):
        pass
