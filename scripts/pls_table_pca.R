library(vegan)

pls_table <- read.csv("C:/Users/darel/Desktop/CAETE-DVM/src/PLS_MAIN/pls_attrs-5000.csv", header = TRUE)
pls_table <- pls_table[,-1]  # Remove the first column if it's an index or identifier


# Perform PCA on the PLS table
table.norm <- decostand(pls_table, method = "standardize")
table.dist <- vegdist(table.norm, method = "euclidean")

pca_result <- prcomp(table.dist)