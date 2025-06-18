library(ggtern)

data <- read.csv("../src/PLS_MAIN/pls_attrs-9999.csv", header = TRUE)

# Create the plot
ggtern(data, aes(x = aleaf, y = awood, z = aroot)) +
  geom_point() +
  labs(
    title = "Allocation coefficients for CAETE",
    x = "aleaf",
    y = "awood",
    z = "aroot"
  ) +
  theme_bw()

# Save the plot
ggsave(
  filename = "../src/PLS_MAIN/ternary_plot_allocation3000.png",
  width = 8,
  height = 6,
  dpi = 300
)
