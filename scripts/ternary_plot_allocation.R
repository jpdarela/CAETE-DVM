library(ggtern)

setwd("./src/PLS_MAIN")
data <- read.csv("pls_attrs-99999.csv", header = TRUE)

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
