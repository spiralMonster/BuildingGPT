class Invalid_Head_Number(Exception):
    def __init__(self,error):
        super().__init__()
        print(error)