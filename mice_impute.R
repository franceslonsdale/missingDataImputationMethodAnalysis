# mice_impute.R ---------------------------------------------------------------
# Called by imputation_methods.py via subprocess.  Reads an incomplete CSV,
# runs mice() with the method vector from the formal plan, and writes m
# completed CSVs back.
#
# Usage:
#   Rscript mice_impute.R <input.csv> <output_prefix> <m> <maxit> <seed>
#
# Writes <output_prefix>_1.csv ... <output_prefix>_<m>.csv
#
# The method vector is fixed to match Deng & Lumley (2024) §3:
#   numeric incomplete columns -> "pmm"
#   bin1  -> "logreg"
#   ord1  -> "polr"
#   Y, norm4, norm6, norm8  -> ""  (fully observed; never imputed)
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(mice)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("Usage: Rscript mice_impute.R <input.csv> <output_prefix> <m> <maxit> <seed>")
}
input_path    <- args[1]
output_prefix <- args[2]
m     <- as.integer(args[3])
maxit <- as.integer(args[4])
seed  <- as.integer(args[5])

set.seed(seed)

df <- read.csv(input_path, stringsAsFactors = FALSE, na.strings = c("NA", ""))

# Enforce correct dtypes -- read.csv will have loaded bin1/ord1 as int
# (with NAs) and we need them as factors for logreg/polr.
df$bin1 <- factor(df$bin1, levels = c("0", "1"))
df$ord1 <- factor(df$ord1, levels = c("0", "1", "2"), ordered = TRUE)

# Method vector: one entry per column, in the order of names(df).
meth <- setNames(rep("", ncol(df)), names(df))
meth["norm1"] <- "pmm"
meth["norm2"] <- "pmm"
meth["norm3"] <- "pmm"
meth["norm5"] <- "pmm"
meth["norm7"] <- "pmm"
meth["bin1"]  <- "logreg"
meth["ord1"]  <- "polr"
# Y, norm4, norm6, norm8 left as "" (never imputed).

fit <- mice(
  df,
  m              = m,
  maxit          = maxit,
  method         = meth,
  printFlag      = FALSE,
  remove.collinear = FALSE
)

for (l in seq_len(m)) {
  comp <- complete(fit, action = l)
  # Convert factors back to integer codes so pandas can read them as ints.
  comp$bin1 <- as.integer(as.character(comp$bin1))
  comp$ord1 <- as.integer(as.character(comp$ord1))
  write.csv(
    comp,
    file = paste0(output_prefix, "_", l, ".csv"),
    row.names = FALSE,
    quote = FALSE
  )
}

cat("mice_impute.R: wrote", m, "files to", output_prefix, "\n")