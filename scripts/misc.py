import matplotlib.pyplot as plt
import ternary
import pandas as pd

from metacommunity import pls_table

# Edite o caminho para o arquivo .csv con as PLSs
table = pls_table.read_pls_table(".pls_attrs-3000.csv")

# Se vc quiser pode pegar apenas um subconjunto dos dados
# Se vc não mudar nada na linha abaixo, todos os dados serão plotados
data = table[:, :]

def ternary_plot(data = data):
    # Create a figure
    figure, tax = ternary.figure(scale=1.0)

    # Plot the data
    points = list(zip(data[0], data[1], data[2]))
    tax.scatter(points, marker='o', color='blue',
                alpha=0.5, label='Dirichlet samples', vmin=0, vmax=1)

    # Set axis labels inside the triangular frame
    tax.left_axis_label("Component 1", offset=0.14)
    tax.right_axis_label("Component 2", offset=0.14)
    tax.bottom_axis_label("Component 3", offset=0.14)

    # Draw boundary and gridlines
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="black", multiple=0.1)

    # Set legend
    tax.legend()

    # Show the plot
    plt.show()

def scatter_matrix(data = data):
    pd.plotting.scatter_matrix(pd.DataFrame(data), figsize=(20, 20))
    plt.savefig("scatter_matrix.png")
