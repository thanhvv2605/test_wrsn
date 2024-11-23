import numpy as np
class RandomController:
    def __init__(self):
        pass

    def make_action(self, id, state, info, wrsn):
        """_summary_

        Args:
            id (_type_): _description_
            state (_type_): _description_
            info (_type_): _description_
            wrsn (_type_): _description_

        Returns:
            _type_: hành động gồm ba thành phần là địa điểm (x,y), thời gian sạc ngẫu nhiên
        """
        if id == None:
            return None
        return np.random.rand(3)

