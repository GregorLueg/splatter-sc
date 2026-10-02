# Generate the R reference fixtures for tests/r_parity.rs.
#
# Usage: Rscript tests/r_parity/make_reference.R [extra R library path]
# Run from the repo root. Writes tests/fixtures/r_parity/.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) .libPaths(c(args[1], .libPaths()))
suppressPackageStartupMessages(library(splatter))

out_root <- file.path("tests", "fixtures", "r_parity")
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

# Full double precision; write.table would round to 15 digits.
fmt <- function(x) {
  if (is.numeric(x)) sprintf("%.17g", x) else as.character(x)
}

write_tsv_gz <- function(df, path) {
  con <- gzfile(path, "w")
  writeLines(paste(names(df), collapse = "\t"), con)
  cols <- lapply(df, fmt)
  writeLines(do.call(paste, c(cols, sep = "\t")), con)
  close(con)
}

write_mat_gz <- function(m, path) {
  con <- gzfile(path, "w")
  m <- as.matrix(m)
  for (i in seq_len(nrow(m))) writeLines(paste(fmt(m[i, ]), collapse = " "), con)
  close(con)
}

write_params <- function(params, method, path) {
  p <- getParams(params, slotNames(params))
  p$method <- method
  writeLines(jsonlite::toJSON(p, auto_unbox = TRUE, digits = NA, pretty = TRUE), path)
}

settings <- list(
  single = list(method = "single", args = list()),
  groups_batches = list(
    method = "groups",
    args = list(
      group.prob = c(0.2, 0.3, 0.5), de.prob = 0.2, de.facLoc = 0.3,
      batch.facLoc = 0.2, batch.facScale = 0.15
    ),
    batch_frac = c(0.4, 0.6)
  ),
  dropout = list(
    method = "single",
    args = list(dropout.type = "experiment", dropout.mid = 1, dropout.shape = -1)
  ),
  lib_norm = list(
    method = "single",
    args = list(lib.norm = TRUE, lib.loc = 20000, lib.scale = 8000)
  )
)

sizes <- list(
  small = list(genes = 100, cells = 200, seed = 1),
  large = list(genes = 2000, cells = 5000, seed = 2)
)

for (name in names(settings)) {
  s <- settings[[name]]
  for (size_name in names(sizes)) {
    sz <- sizes[[size_name]]
    frac <- if (is.null(s$batch_frac)) 1 else s$batch_frac
    batch_cells <- round(sz$cells * frac)
    params <- do.call(newSplatParams, c(
      list(nGenes = sz$genes, batchCells = batch_cells, seed = sz$seed),
      s$args
    ))
    sim <- splatSimulate(params, method = s$method, verbose = FALSE)

    out <- file.path(out_root, name, size_name)
    dir.create(out, recursive = TRUE, showWarnings = FALSE)
    write_params(params, s$method, file.path(out, "params.json"))
    write_params(splatter:::expandParams(params), s$method, file.path(out, "params_expanded.json"))

    a <- SummarizedExperiment::assays(sim)
    rd <- as.data.frame(SummarizedExperiment::rowData(sim))
    # splatter does not store the per-gene sqrt(bcv.df / chisq); recover it
    # from cell 1 so the Rust side can run on R's exact gene truth.
    rd$BCVChiFac <- a$BCV[, 1] /
      (getParam(params, "bcv.common") + 1 / sqrt(a$BaseCellMeans[, 1]))
    cd <- as.data.frame(SummarizedExperiment::colData(sim))
    if (!is.null(cd$Group)) cd$Group <- as.character(cd$Group)
    write_tsv_gz(rd, file.path(out, "rowData.tsv.gz"))
    write_tsv_gz(cd, file.path(out, "colData.tsv.gz"))

    if (size_name == "small") {
      write_mat_gz(a$BaseCellMeans, file.path(out, "BaseCellMeans.txt.gz"))
      write_mat_gz(a$BCV, file.path(out, "BCV.txt.gz"))
    } else {
      counts <- as.matrix(SummarizedExperiment::assay(sim, "counts"))
      gs <- data.frame(
        Gene = rownames(counts),
        Mean = rowMeans(counts),
        Var = apply(counts, 1, var),
        ZeroFrac = rowMeans(counts == 0)
      )
      if (!is.null(cd$Group)) {
        for (g in sort(unique(cd$Group))) {
          sub <- counts[, cd$Group == g, drop = FALSE]
          gs[[paste0("Mean", g)]] <- rowMeans(sub)
          gs[[paste0("Var", g)]] <- apply(sub, 1, var)
        }
      }
      cs <- data.frame(
        Cell = colnames(counts),
        Total = colSums(counts),
        Detected = colSums(counts > 0)
      )
      write_tsv_gz(gs, file.path(out, "gene_summary.tsv.gz"))
      write_tsv_gz(cs, file.path(out, "cell_summary.tsv.gz"))
    }
  }
}

# Parameter sets R must accept or reject; the Rust port has to agree.
cases <- list(
  valid_groups = list(group.prob = c(0.2, 0.8)),
  valid_batches = list(batchCells = c(10, 20), batch.facLoc = c(0.1, 0.2)),
  batch_loc_length = list(batchCells = c(10, 10), batch.facLoc = c(0.1, 0.2, 0.3)),
  de_prob_length = list(group.prob = c(0.5, 0.5), de.prob = c(0.1, 0.2, 0.3)),
  out_prob_range = list(out.prob = 1.5),
  lib_scale_negative = list(lib.scale = -1),
  de_down_prob_range = list(de.downProb = 2),
  bcv_common_negative = list(bcv.common = -0.1),
  n_genes_zero = list(nGenes = 0),
  batch_scale_negative = list(batchCells = c(10, 10), batch.facScale = -1),
  dropout_experiment_length = list(
    dropout.mid = c(1, 2), dropout.shape = c(1, 2), dropout.type = "experiment"
  ),
  dropout_batch_length = list(batchCells = c(10, 10), dropout.type = "batch")
)
res <- lapply(names(cases), function(n) {
  ok <- tryCatch(
    {
      p <- suppressWarnings(do.call(newSplatParams, cases[[n]]))
      validObject(p)
      TRUE
    },
    error = function(e) FALSE
  )
  list(name = n, params = cases[[n]], valid = ok)
})
writeLines(
  jsonlite::toJSON(res, auto_unbox = TRUE, digits = NA, pretty = TRUE),
  file.path(out_root, "validity_cases.json")
)

writeLines(
  c(
    paste("splatter", as.character(packageVersion("splatter"))),
    paste(R.version.string)
  ),
  file.path(out_root, "VERSION")
)
