
class budget_output:
    """ Helper class to store the output of the daily_budget function.
    """
    def __init__(self, *args):
        fields = ["evavg", "epavg", "phavg", "aravg", "nppavg",
                  "laiavg", "rcavg", "f5avg", "rmavg", "rgavg",
                  "cleafavg_pft", "cawoodavg_pft", "cfrootavg_pft",
                  "stodbg", "ocpavg", "wueavg", "cueavg", "c_defavg",
                  "vcmax", "specific_la", "nupt", "pupt", "litter_l",
                  "cwd", "litter_fr", "npp2pay", "lnc", "limitation_status",
                  "uptk_strat", 'cp', 'c_cost_cwm']

        for field, value in zip(fields, args):
            setattr(self, field, value)

class daily_out_manager:
    "write a class that manages n instances of the above class budget_output"
    def __init__(self) -> None:
        pass


class netcdf_output:
    pass