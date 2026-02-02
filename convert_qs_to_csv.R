# Load necessary libraries
library(qs)
library(SingleCellExperiment)

# Set working directory where the .qs files are located
setwd("data/stamp_fig1_samples")

# List of sample base names
samples <- c(
  "GSM8814934_Stamp_C_02_SKBR3",
  "GSM8814931_Stamp_C_02_LnCAP",
  "GSM8814932_Stamp_C_02_MCFF7",
  "GSM8814933_Stamp_C_02_MIX"
)

# Loop through each sample
for (s in samples) {
  qs_file <- paste0(s, ".qs")

  # Read in the SingleCellExperiment object
  obj <- qread(qs_file)

  # Extract metadata and counts
  metadata <- as.data.frame(colData(obj))
  #counts <- as.matrix(assay(obj, "counts"))

  # Write outputs
  write.csv(metadata, paste0(s, "_metadata.csv"), row.names = TRUE)
  #write.csv(counts, paste0(s, "_counts.csv"), row.names = TRUE)

  cat("✅ Written:", s, "_metadata.csv and _counts.csv\n")
}

