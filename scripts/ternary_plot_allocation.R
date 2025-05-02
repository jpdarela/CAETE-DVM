library(ggtern)

data <- read.csv("../src/PLS_MAIN/pls_attrs-3000.csv", header = TRUE)

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
