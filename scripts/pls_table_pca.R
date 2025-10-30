library(vegan)

pls_table <- read.csv("C:/Users/darel/Desktop/CAETE-DVM/src/PLS_MAIN/pls_attrs-500.csv", header = TRUE)
pls_table <- pls_table[,-1]  # Remove the first column if it's an index or identifier


# Perform PCA on the PLS table
table.norm <- decostand(pls_table, method = "standardize")

# Calculate only the first 2 PCs
pca_result <- prcomp(table.norm, rank. = 2)
