from pathlib import Path
import numpy as np
from plsgen import table_gen
from caete_module import global_par as gp

ntraits = gp.ntraits
comm_npls = gp.npls

class pls_table:
    def __init__(self, array:np.ndarray) -> None:
        self.table = array
        self.npls = self.table.shape[1]
        self.ntraits = self.table.shape[0]
        self.shape = self.table.shape
        self.id = np.arange(self.npls)

    def __len__(self):
        return self.npls

    def __getitem__(self, index):
        return self.table[:,index]

    def get_random_pls(self):
        id = np.random.randint(0, self.npls-1)
        return id, self.table[:,id]

    def create_npls_table(self, comm_npls):
        added = []
        table = np.zeros((self.ntraits, comm_npls), dtype=np.float32)
        idx = 0
        while len(added) < comm_npls:
            id, pls = self.get_random_pls()

            if id not in added:
                table[:,idx] = pls
                added.append(id)
                idx += 1
        return id, table

        # return np.array([self.get_random_pls() for _ in range(comm_npls)]).T


class community:

    def __init__(self, pls_table) -> None:
        self.pls_table = pls_table
        self.abundance = np.zeros(gp.npls, dtype=np.float64)
        self.alive = np.ones(gp.npls, dtype=bool)
        self.living_pls = {}



class metacommunity:

    def __init__(self, n:int, pls_table:pls_table) -> None:

        self.communities = {}
        self.pls_table = pls_table

        for i in range(n):
            self.communities[i] = community(self.pls_table.create_npls_table(gp.npls))

    def __getitem__(self, index):
        return self.communities[index]

    def __len__(self):
        return len(self.communities)



if __name__ == "__main__":
    main_table = table_gen(NPLS=20, fpath=Path("./TEST"))
    table_obj = pls_table(main_table)

    comm = community(table_obj.create_npls_table(10))

    # mt = metacommunity(5, table_obj)
    # # print(mt[0].pls_table.table)
    # # print(mt[0].pls_table.id)
    # # print(mt[0].pls_table.get_random_pls())
    # # print(mt[0].pls_table.create_npls_table(10))
    # # print(len(mt))