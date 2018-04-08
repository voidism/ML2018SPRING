import time

class counter:
    def __init__(self,epoch,update_rate=50,title = ""):
        self.title = title
        self.start_time = time.time()
        self.iteration = epoch
        self.rate = update_rate
        self.rem_time = 0
        self.pass_time = 0
        self.cost = 0
    
    def set_start(self):
        self.start_time = time.time()
    
    def set_total(self,epoch):
        self.iteration = epoch

    def flush(self,j,cost = 0):
        if j == 0 and self.title != "":
            print("<=== ["+self.title+"] ===>")
        elif j % self.rate == 1:
            self.pass_time = time.time() - self.start_time
            self.rem_time = self.pass_time * (self.iteration - j) / j
        print(
            chr(13) + "|" + "=" * (50 * j // self.iteration
            ) + ">" + " " * (50 * (self.iteration - j) // self.iteration
            ) + "| " + str(
                round(100 * j / self.iteration, 1)) + "%",
            "\tave cost: "+str(round(cost,2)) if cost!= 0 else "", 
            "\tremain:",round(self.rem_time,0),"s ",
            sep=' ', end = '', flush = True)
        if j == self.iteration-1:
            print("")