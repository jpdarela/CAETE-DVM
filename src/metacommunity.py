from pathlib import Path
import numpy as np
from caete import read_pls_table
from caete_module import global_par as gp
from caete_module import photo as m



class pls_table:
    """ Interface for the main table of PLSs. Random subsamples without replacement
        are taken from this table to create communities. The main table should've
        a large number of PLSs (npls ~ 20000, for example).
    """
    def __init__(self, array:np.ndarray) -> None:
        """Provides an interface for the main table of PLSs.

        Args:
            array (np.ndarray(ntraits,npls)): PLS table.
        """
        self.table = array
        self.npls = self.table.shape[1]
        self.ntraits = self.table.shape[0]
        self.shape = self.table.shape
        self.id = np.arange(self.npls)


    def __len__(self):
        return self.npls


    def __getitem__(self, index):
        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.table[:,index]


    def get_random_pls(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        id = np.random.randint(0, self.npls-1)
        return id, self.table[:,id]


    def create_npls_table(self, comm_npls):
        """_summary_

        Args:
            comm_npls (_type_): _description_

        Returns:
            _type_: _description_
        """
        idx = np.random.randint(0, self.npls-1, comm_npls)
        table = self.table[:,idx]
        return idx, table


class community:
    """Represents a community of plants. Instances of this class are used to create metacommunities."""

    def __init__(self, pls_table:np.ndarray) -> None:
        """Starts an assembly of plants.

        Args:
            pls_table (np.ndarray): A table of PLSs. Each column is a PLS of a plant.
            The table is a 2D numpy array with rank(ntraits, npls).
        """
        self.id, self.pls_table = pls_table
        self.npls = self.pls_table.shape[1]
        self.shape = self.pls_table.shape
        self.alive = np.ones(gp.npls, dtype=bool)

        # BIOMASS_STATE
        self.vp_cleaf = np.random.uniform(0.3,0.4,self.npls)#np.zeros(shape=(npls,), order='F') + 0.1
        self.vp_croot = np.random.uniform(0.3,0.4,self.npls)#np.zeros(shape=(npls,), order='F') + 0.1
        self.vp_cwood = np.random.uniform(5.0,6.0,self.npls)#np.zeros(shape=(npls,), order='F') + 0.1

        self.vp_cwood[self.pls_table[6,:] == 0.0] = 0.0

        self.vp_ocp, b, c, d = m.pft_area_frac(self.vp_cleaf, self.vp_croot,
                                               self.vp_cwood, self.pls_table[6, :])
        del b, c, d
        self.vp_lsid = np.where(self.vp_ocp > 0.0)[0]
        self.ls = self.vp_lsid.size
        self.vp_sto = np.zeros(shape=(3, self.npls), order='F')

    def __getitem__(self, index:int):
        """Gets a PLS (1D array) given index.

        Args:
            index (int): _description_

        Returns:
            1D np.ndarray: PLS of the plant at the given index.
        """
        return self.pls_table[:,index]

    def __len__(self):
        return self.shape[1]


class metacommunity:

    def __init__(self, n:int, main_table:pls_table) -> None:

        self.communities = {}
        self.pls_table = main_table

        for i in range(n):
            self.communities[i] = community(self.pls_table.create_npls_table(gp.npls))

    def __getitem__(self, index):
        return self.communities[index]

    def __len__(self):
        return len(self.communities)

    def __iter__(self):
        return iter(self.communities.values())


def main():
    main_table = read_pls_table(Path("./PLS_MAIN/pls_attrs-25000.csv"))
    table_obj = pls_table(main_table)

    # comm = community(table_obj.create_npls_table(10))

    return metacommunity(99, table_obj)
    # # print(mt[0].pls_table.table)
    # # print(mt[0].pls_table.id)
    # # print(mt[0].pls_table.get_random_pls())
    # # print(mt[0].pls_table.create_npls_table(10))
    # # print(len(mt))


if __name__ == "__main__":
    mt = main()
