from defense.defense_base import Defender


class MIDefender(Defender):
    def __init__(self, **kwargs):
        super().__init__()

    def pre_epoch(self, **kwargs):
        pass

    def post_epoch(self, **kwargs):
        pass


class EmptyMIDefender(MIDefender):
    name = "EmptyMIDefender"

    def pre_epoch(self, **kwargs):
        pass

    def post_epoch(self, **kwargs):
        pass
