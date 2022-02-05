sla_calc <- function(leaf_residence_time)
{
    # Residence_time in years
    n_tau_leaf = (leaf_residence_time - 0.08333333)/(8.33333333 - 0.08333333)

    tl0 = (365.242 / 12.0) * (10.0 ^ (2.0*n_tau_leaf))

    sla = (3.0 * (365.242 / tl0) ^ (-0.46))
    sla # Resultado em m²kg⁻¹ (Metros quadrados por kilograma de C)
}
