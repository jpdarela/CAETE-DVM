import matplotlib.pyplot as plt

# Create a new figure
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

# LaTeX-formatted equation
equation = (
    r"$f_5 = \max\left(0,\ 1 - \exp\left(-0.1 \cdot \frac{pt}{d}\right)\right)$"
    "\n\n"
    r"$pt = csru \cdot (cf_{root} \times 1000) \cdot wa$"
    "\n"
    r"$wa = \frac{w}{w_{max}}$"
    "\n"
    r"$d = \frac{ep \cdot \alpha_m}{1 + \frac{g_m}{g_c}}$"
    "\n"
    r"$g_c = \frac{41.67}{rc}$"
)

# Add the equation text to the figure
plt.text(0.5, 0.5, equation, fontsize=20, ha='center', va='center', family='serif')

# Save as image
plt.savefig("water_stress_equation.png", bbox_inches='tight', dpi=200)
plt.show()