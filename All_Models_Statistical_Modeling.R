###############################################################################
# Statistical Modeling of Physical Activity and Gait Kinematics
#
# Contents
#   1. Setup and utilities
#   2. Regression modeling (PCR and PLS)
#   3. PCA-based classification with decision trees (C5.0)
#   4. PCA + LDA classification
#   5. Varimax loading visualization
#   6. Example PC group comparison
#   7. Example clinical scatterplot
#   8. Mediation analysis
###############################################################################

############################
# 1. Setup and utilities   #
############################

suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(pls)
  library(ggplot2)
  library(ggfortify)
  library(devtools)
  library(C50)
  library(MASS)
  library(ROCR)
  library(factoextra)
  library(mediation)
})  # [file:2]

set.seed(1)

# Helper: standard train–test split
train_test_split <- function(data, response, prop = 0.8) {
  idx <- createDataPartition(data[[response]], p = prop, list = FALSE)
  list(
    train = data[idx, , drop = FALSE],
    test  = data[-idx, , drop = FALSE]
  )
}  # [file:2]

# Helper: caret regression evaluation
evaluate_regression <- function(pred, truth) {
  data.frame(
    RMSE    = caret::RMSE(pred, truth),
    Rsquare = caret::R2(pred, truth)
  )
}  # [file:2]

# Helper: caret classification evaluation
evaluate_classification <- function(pred, truth) {
  cm <- confusionMatrix(factor(pred), factor(truth))
  list(
    confusion_matrix = cm$table,
    overall          = cm$overall,
    by_class         = cm$byClass
  )
}  # [file:2]


#############################################
# 2. Regression modeling (PCR and PLS)     #
#############################################

run_pcr_model <- function(
  file       = "PASE_study_Kinematics_PCR.csv",
  response   = "PASE",
  sep        = ";",
  dec        = ",",
  na_strings = c("", " ", "NA"),
  cv_folds   = 10,
  tune_len   = 10
) {
  message("Loading data for PCR ...")
  df <- read.csv(file, sep = sep, dec = dec, na.strings = na_strings)

  split <- train_test_split(df, response = response, prop = 0.8)
  train.data <- split$train
  test.data  <- split$test

  set.seed(1)
  message("Fitting PCR model ...")
  model <- train(
    reformulate(".", response),
    data      = train.data,
    method    = "pcr",
    scale     = TRUE,
    trControl = trainControl(method = "cv", number = cv_folds),
    tuneLength = tune_len
  )

  print(model)
  plot(model)
  cat("\nBest tuning parameters:\n")
  print(model$bestTune)
  cat("\nFinal model summary:\n")
  print(summary(model$finalModel))

  message("Evaluating PCR on test set ...")
  predictions <- predict(model, newdata = test.data)
  perf <- evaluate_regression(predictions, test.data[[response]])
  print(perf)

  invisible(list(model = model, performance = perf))
}  # [file:2]


run_pls_model <- function(
  file       = "PASE_study_Kinematics_PCR.csv",
  response   = "PASE",
  sep        = ";",
  dec        = ",",
  na_strings = c("", " ", "NA"),
  cv_folds   = 10,
  tune_len   = 10
) {
  message("Loading data for PLS ...")
  df <- read.csv(file, sep = sep, dec = dec, na.strings = na_strings)

  split <- train_test_split(df, response = response, prop = 0.8)
  train.data <- split$train
  test.data  <- split$test

  set.seed(123)
  message("Fitting PLS model ...")
  model <- train(
    reformulate(".", response),
    data      = train.data,
    method    = "pls",
    scale     = TRUE,
    trControl = trainControl(method = "cv", number = cv_folds),
    tuneLength = tune_len
  )

  print(model)
  plot(model)
  cat("\nBest tuning parameters:\n")
  print(model$bestTune)
  cat("\nFinal model summary:\n")
  print(summary(model$finalModel))

  message("Evaluating PLS on test set ...")
  predictions <- predict(model, newdata = test.data)
  perf <- evaluate_regression(predictions, test.data[[response]])
  print(perf)

  invisible(list(model = model, performance = perf))
}  # [file:2]


#########################################################
# 3. PCA-based classification with C5.0 decision tree   #
#########################################################

pca_tree_classification <- function(
  file       = "PASE_PCA_short.csv",
  label_col  = "PA",
  sep        = ";",
  dec        = ",",
  na_strings = c("", " ", "NA")
) {
  message("Loading PCA classification dataset ...")
  df <- read.csv(file, sep = sep, dec = dec, na.strings = na_strings)
  df[[label_col]] <- as.factor(df[[label_col]])

  # Use first 54 columns as numeric predictors (as in original script)
  predictors <- df[, 1:54]

  message("Plotting PC1 vs PC2 colored by label ...")
  p <- autoplot(prcomp(predictors),
                data   = df,
                colour = label_col,
                shape  = label_col) +
    geom_point(aes(size = 40, colour = .data[[label_col]], shape = .data[[label_col]]),
               show.legend = FALSE) +
    theme_minimal()
  print(p)

  message("Pre-processing with PCA (caret::preProcess) ...")
  pca_model <- preProcess(df[, -which(names(df) == label_col)], method = "pca")
  PC <- predict(pca_model, df[, -which(names(df) == label_col)])

  str(PC)
  cat("\nPCA loadings:\n")
  print(pca_model$rotation)

  # Combine PCs with label
  tr <- cbind(PC, label = df[[label_col]])
  tr$label <- as.factor(tr$label)

  message("Train/test split for C5.0 ...")
  idx <- createDataPartition(tr$label, p = 0.8, list = FALSE)
  train.data <- tr[idx, , drop = FALSE]
  valid.data <- tr[-idx, , drop = FALSE]

  message("Fitting C5.0 tree ...")
  tree_model <- C5.0(label ~ ., data = train.data)
  print(summary(tree_model))

  message("Evaluating tree on validation set ...")
  pred <- predict(tree_model, newdata = valid.data[, !names(valid.data) %in% "label"])
  acc <- mean(pred == valid.data$label)
  cat(sprintf("\nValidation accuracy: %.3f\n", acc))

  message("Plotting decision tree ...")
  plot(tree_model, colours(distinct = TRUE))

  invisible(list(
    pca_model = pca_model,
    tree_model = tree_model,
    accuracy = acc
  ))
}  # [file:2]


########################################
# 4. PCA + LDA classification pipeline #
########################################

pca_lda_classification <- function(
  file       = "PASE_PCA_short.csv",
  label_col  = "PA",
  sep        = ";",
  dec        = ",",
  na_strings = c("", " ", "NA"),
  n_pcs      = 10,
  train_prop = 0.70
) {
  message("Loading dataset for PCA + LDA ...")
  df <- read.csv(file, sep = sep, dec = dec, na.strings = na_strings)
  df[[label_col]] <- as.factor(df[[label_col]])

  message("Running PCA on kinematic variables ...")
  # Use first 44 predictors as in original script
  pca <- prcomp(df[, 1:44], center = TRUE, scale. = TRUE)
  print(summary(pca))

  cat("\nFirst 10 loadings:\n")
  print(pca$rotation[, 1:n_pcs])

  # Scree and cumulative variance plots (optional)
  screeplot(pca, type = "l", npcs = 12,
            main = "Screeplot of the first 12 PCs")
  abline(h = 1, col = "red", lty = 5)
  legend("topright", legend = "Eigenvalue = 1",
         col = "red", lty = 5, cex = 0.6)

  cumpro <- cumsum(pca$sdev^2 / sum(pca$sdev^2))
  plot(cumpro[1:12],
       xlab = "PC #",
       ylab = "Cumulative proportion of explained variance",
       main = "Cumulative variance plot")
  abline(v = n_pcs, col = "blue", lty = 5)
  legend("topleft", legend = sprintf("Cut-off @ PC%d", n_pcs),
         col = "blue", lty = 5, cex = 0.6)

  message("2D PCA plot (PC1 vs PC2) ...")
  plot(pca$x[, 1], pca$x[, 2],
       xlab = "PC1",
       ylab = "PC2",
       main = "PC1 / PC2 plot")

  fviz_pca_ind(
    pca,
    geom.ind  = "point",
    pointshape = 21,
    pointsize  = 4,
    fill.ind   = df[[label_col]],
    col.ind    = "black",
    palette    = "jco",
    addEllipses = TRUE,
    label      = "var",
    col.var    = "black",
    repel      = TRUE,
    legend.title = "Physical Activity Group"
  ) +
    ggtitle("2D PCA-plot based on the 44 kinematic features") +
    theme(plot.title = element_text(hjust = 0.5))

  message("Building PC dataset for LDA ...")
  pcs <- pca$x[, 1:n_pcs]
  pcs_df <- cbind(pcs, PA = df[[label_col]])

  set.seed(4)
  smp_size <- floor(train_prop * nrow(pcs_df))
  idx <- sample(seq_len(nrow(pcs_df)), size = smp_size)

  train.df <- as.data.frame(pcs_df[idx, , drop = FALSE])
  test.df  <- as.data.frame(pcs_df[-idx, , drop = FALSE])

  message("Fitting LDA model ...")
  lda_formula <- as.formula(
    paste("PA ~", paste(colnames(pcs_df)[1:n_pcs], collapse = " + "))
  )
  lda_model <- lda(lda_formula, data = train.df)

  pred <- predict(lda_model, newdata = test.df)
  cat("\nLDA model:\n")
  print(lda_model)
  cat("\nLDA prediction summary:\n")
  print(pred)

  cat("\nConfusion matrix (LDA):\n")
  print(table(pred$class, test.df$PA))

  cm <- confusionMatrix(
    data      = factor(pred$class),
    reference = factor(test.df$PA)
  )
  print(cm)

  invisible(list(
    pca      = pca,
    lda_model = lda_model,
    confusion = cm
  ))
}  # [file:2]


#########################################
# 5. Varimax loading visualization      #
#########################################

plot_varimax_heatmap <- function(
  file_varimax        = "Varimax.csv",
  trimmed_output      = "Varimaxtable_Trimmed.csv",
  trimmed_input_plot  = "Varimaxtable_Trimmed copy.csv"
) {
  message("Preparing varimax heatmap ...")

  varimax_table <- read.csv(file_varimax, sep = ";", dec = ".", header = TRUE)
  varimax_table[
    varimax_table < 0.4 & varimax_table > -0.4
  ] <- ""

  write.csv(varimax_table, trimmed_output, row.names = FALSE)

  varimax_trimmed <- read.csv(trimmed_input_plot,
                              sep = ";", dec = ".", header = TRUE)
  varimax_trimmed$Component <- factor(
    varimax_trimmed$Component,
    levels = paste0("RC", 1:10)
  )

  p <- ggplot(varimax_trimmed,
              aes(x = Component, y = Variable, fill = Value)) +
    geom_tile() +
    geom_text(aes(label = Value)) +
    scale_fill_continuous(
      low      = "#018571",
      high     = "#FC717F",
      na.value = "white"
    ) +
    theme_minimal()

  print(p)
  invisible(p)
}  # [file:2]


########################################################
# 6. Example PC comparison between two patient groups  #
########################################################

compare_pc_groups <- function(PC, group_index_1, group_index_2, pc_col = 10) {
  message("Paired t-test on selected PC ...")
  res <- t.test(PC[group_index_1, pc_col],
                PC[group_index_2, pc_col],
                paired = TRUE)
  print(res)
  invisible(res)
}  # [file:2]


#####################################
# 7. Example clinical scatter plot  #
#####################################

plot_clinical_scatter <- function(Longitudinal) {
  ggplot(Longitudinal,
         aes(x = PASE, y = Rigidity, color = Rigidity)) +
    geom_smooth(colour = "bisque2",
                fill   = "aquamarine3",
                method = "loess",
                alpha  = 0.1) +
    geom_point(size = 10, alpha = 0.8, show.legend = FALSE) +
    scale_shape(guide = FALSE) +
    theme(
      legend.position   = c(.9, .75),
      legend.spacing.y  = unit(1, "mm"),
      panel.border      = element_rect(colour = "black", fill = NA),
      legend.background = element_blank(),
      legend.box.background = element_rect(colour = "black")
    ) +
    labs(
      x = "PASE Score",
      y = "Rigidity Score"
    ) +
    scale_color_gradient(low = "bisque2", high = "aquamarine3") +
    theme_minimal(base_size = 20)
}  # [file:2]


###########################
# 8. Mediation analysis   #
###########################

run_mediation <- function(
  file       = "dfMediation.csv",
  sep        = ";",
  dec        = ",",
  na_strings = c("", " ", "NA"),
  treat      = "PASE",
  mediator   = "UPDRS.III",
  outcome    = "PDQ.8",
  n_boot     = 1000
) {
  message("Running mediation analysis ...")
  df <- read.csv(file, sep = sep, dec = dec, na.strings = na_strings)
  df <- na.omit(df)

  fit_total <- lm(reformulate(treat, outcome), data = df)
  fit_med   <- lm(reformulate(treat, mediator), data = df)
  fit_dv    <- lm(reformulate(c(treat, mediator), outcome), data = df)

  cat("\nTotal effect model:\n")
  print(summary(fit_total))
  cat("\nMediator model:\n")
  print(summary(fit_med))
  cat("\nOutcome model (with mediator):\n")
  print(summary(fit_dv))

  set.seed(1)
  med_res <- mediate(
    model.m = fit_med,
    model.y = fit_dv,
    treat   = treat,
    mediator = mediator,
    boot    = TRUE,
    sims    = n_boot
  )

  cat("\nMediation results:\n")
  print(summary(med_res))

  invisible(list(
    fit_total = fit_total,
    fit_med   = fit_med,
    fit_dv    = fit_dv,
    med_res   = med_res
  ))
}  # [file:2]


###############################################################################
# Example usage (uncomment the calls you need)
###############################################################################
# pcr_results  <- run_pcr_model()
# pls_results  <- run_pls_model()
# tree_results <- pca_tree_classification()
# lda_results  <- pca_lda_classification()
# plot_varimax_heatmap()
# med_results  <- run_mediation()
