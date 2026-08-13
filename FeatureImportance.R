# Load required packages
library(shapviz)
library(xgboost)
library(caret)
library(pROC)
library(ROCR) 
library(tibble)
library(ROCit)
library(readxl)
library(haven)
library(ggplot2)
library(gridExtra)
library(iml)
library(showtext)
library(svglite)
library(dplyr)
library(PRROC)

# Set working directory and read data
setwd("E:/zcx/healthyaging/2011-new") 
data <- read_dta("Analysis_0.dta")

# Load Calibri font for plots
font_add('Calibri','/C:/Windows/Fonts/Calibri.ttf')
showtext_auto()
windowsFonts(A = windowsFont("Calibri"))

# Filter year and mobile user condition
data <- data[data$year == 2011, ]
data <- data[data$mobileuser3 == 2, ]

# Extract grouping variable (city)
if(!"city" %in% names(data)) stop("Column 'city' not found in data.")
city_vec <- data$city

# Extract variables
healthyaging <- data$healthyaging
density      <- data$density
height       <- data$height
agr_density  <- data$agr_area      # annual growth rate of density
agr_height   <- data$agr_height    # annual growth rate of height
per_volume   <- data$pervolume

# Covariates
age          <- data$age
gender       <- data$gender
rural        <- data$rural
education    <- data$education
marriage     <- data$marriage
perconsume   <- data$perconsume
drink        <- data$drink
smoke        <- data$smoke
pop_density  <- data$pop_density
per_gdp      <- data$per_gdp

# Build modelling dataset
datab <- data.frame(
  healthyaging, per_volume, density, height, agr_density, agr_height,
  pop_density, per_gdp, age, gender, rural, education, marriage,
  drink, smoke, perconsume
)

# Remove rows with missing outcome
valid_idx <- !is.na(datab$healthyaging)
datab     <- datab[valid_idx, ]
city_vec  <- city_vec[valid_idx]

# Use all data as training set
traindata <- datab
str(traindata)
table(traindata$healthyaging)

# Create xgb.DMatrix
dtrain <- xgb.DMatrix(data = as.matrix(traindata[, -1]), 
                      label = traindata$healthyaging)

# Compute scale_pos_weight
pos_count <- sum(traindata$healthyaging == 1)
neg_count <- sum(traindata$healthyaging == 0)
scale_pos_weight_value <- neg_count / pos_count

# Create city‑based cross‑validation folds (5‑fold)
train_folds <- groupKFold(city_vec, k = 5)
test_folds  <- lapply(train_folds, function(train_idx) 
  setdiff(seq_len(nrow(traindata)), train_idx))

# Parameter grid
param_grid <- expand.grid(
  max_depth        = c(3,4,5,6,7),
  eta              = c(0.05, 0.1, 0.2, 0.3),
  min_child_weight = c(3, 5, 7, 9),
  gamma            = c(0, 0.2, 0.5, 0.7, 1),
  lambda           = c(1, 10),
  alpha            = c(0, 1),
  colsample_bytree = c(0.7),
  subsample        = c(0.7),
  scale_pos_weight = c(scale_pos_weight_value)
)

cat("Starting grid search over", nrow(param_grid), "combinations...\n")

# Store results
results <- data.frame(
  max_depth = numeric(), eta = numeric(), min_child_weight = numeric(),
  gamma = numeric(), subsample = numeric(), colsample_bytree = numeric(),
  scale_pos_weight = numeric(), lambda = numeric(), alpha = numeric(),
  best_nrounds = numeric(), best_auc = numeric(),
  best_auc_ci_low = numeric(), best_auc_ci_high = numeric()
)

# Grid search loop
for (i in 1:nrow(param_grid)) {
  params <- list(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    max_depth        = param_grid$max_depth[i],
    eta              = param_grid$eta[i],
    min_child_weight = param_grid$min_child_weight[i],
    gamma            = param_grid$gamma[i],
    subsample        = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i],
    scale_pos_weight = param_grid$scale_pos_weight[i],
    lambda           = param_grid$lambda[i],
    alpha            = param_grid$alpha[i]
  )
  
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    folds = test_folds,
    nrounds = 300,
    early_stopping_rounds = 30,
    verbose = 0
  )
  
  best_iter <- cv$best_iteration
  best_mean <- cv$evaluation_log$test_auc_mean[best_iter]
  best_std  <- cv$evaluation_log$test_auc_std[best_iter]
  
  ci_low  <- best_mean - 1.96 * best_std
  ci_high <- best_mean + 1.96 * best_std
  
  results <- rbind(results, data.frame(
    max_depth        = params$max_depth,
    eta              = params$eta,
    min_child_weight = params$min_child_weight,
    gamma            = params$gamma,
    subsample        = params$subsample,
    colsample_bytree = params$colsample_bytree,
    scale_pos_weight = params$scale_pos_weight,
    lambda           = params$lambda,
    alpha            = params$alpha,
    best_nrounds     = best_iter,
    best_auc         = best_mean,
    best_auc_ci_low  = ci_low,
    best_auc_ci_high = ci_high
  ))
}

# Best parameters
best_params <- results %>% filter(best_auc == max(best_auc))
cat("Best parameters (group CV):\n")
print(best_params)
cat("\nOptimal AUC:", round(best_params$best_auc, 3),
    " (95% CI: ", round(best_params$best_auc_ci_low, 3),
    " - ", round(best_params$best_auc_ci_high, 3), ")\n", sep = "")

# Train final model
model_xgboost <- xgboost(
  data = as.matrix(traindata[, -1]),
  label = traindata$healthyaging,
  max_depth        = best_params$max_depth,
  eta              = best_params$eta,
  subsample        = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  gamma            = best_params$gamma,
  scale_pos_weight = best_params$scale_pos_weight,
  lambda           = best_params$lambda,
  alpha            = best_params$alpha,
  nrounds          = best_params$best_nrounds,
  objective        = "binary:logistic"
)

# Predictions
traindata$pred <- predict(model_xgboost, as.matrix(traindata[, -1]))

# ROC and AUC on training set
ROC_train <- roc(traindata$healthyaging, traindata$pred)
specificity <- ROC_train$specificities
sensitivity <- ROC_train$sensitivities
one_minus_specificity <- 1 - specificity

plot(one_minus_specificity, sensitivity, type = "l", col = "blue", lwd = 2,
     xlab = "1 - Specificity", ylab = "Sensitivity", main = "Training ROC Curve")
abline(a = 0, b = 1, lty = 2, col = "red")
CI_train <- ci(ROC_train)
AUC_CI_train <- paste0("AUC=", round(CI_train[2], 3), 
                       ", 95% CI (", round(CI_train[1], 3), " - ", round(CI_train[3], 3), ")")
text(0.6, 0.4, AUC_CI_train, col = "blue")
cat("Training set:", AUC_CI_train, "\n")

# PR curve and AP (with bootstrap CI)
bootstrap_ap_ci <- function(labels, preds, n_boot = 1000, conf = 0.95) {
  aps <- numeric(n_boot)
  n <- length(labels)
  for (i in 1:n_boot) {
    idx <- sample(1:n, size = n, replace = TRUE)
    boot_labels <- labels[idx]
    boot_preds <- preds[idx]
    pr <- pr.curve(scores.class0 = boot_preds[boot_labels == 1],
                   scores.class1 = boot_preds[boot_labels == 0], curve = FALSE)
    aps[i] <- pr$auc.integral
  }
  alpha <- (1 - conf) / 2
  ci_lower <- quantile(aps, alpha, na.rm = TRUE)
  ci_upper <- quantile(aps, 1 - alpha, na.rm = TRUE)
  return(list(mean_ap = mean(aps), ci = c(ci_lower, ci_upper)))
}

pr_train <- pr.curve(scores.class0 = traindata$pred[traindata$healthyaging == 1],
                     scores.class1 = traindata$pred[traindata$healthyaging == 0], curve = TRUE)
cat("Training AP:", pr_train$auc.integral, "\n")
ap_ci_train <- bootstrap_ap_ci(traindata$healthyaging, traindata$pred, n_boot = 1000)
cat("Training AP 95% CI:", round(ap_ci_train$ci[1], 3), "-", round(ap_ci_train$ci[2], 3), "\n")
plot(pr_train, main = "Precision-Recall Curve", col = "blue", lwd = 2)

# Confusion matrix
threshold <- coords(ROC_train, "best", ret = "threshold", transpose = FALSE)$threshold
cat("Best threshold:", threshold, "\n")
train_pred_label <- ifelse(traindata$pred > threshold, 1, 0)
train_conf_matrix <- confusionMatrix(as.factor(train_pred_label), as.factor(traindata$healthyaging))
cat("Training confusion matrix:\n")
print(train_conf_matrix)
fourfoldplot(train_conf_matrix$table, color = c("#FF9999", "#99CCFF"), 
             main = "Training Confusion Matrix")

# Feature importance (Gain) and SHAP
custom_theme <- theme_minimal(base_family = "Calibri") +
  theme(
    plot.title = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 44, color = "black"),
    axis.text.y = element_text(size = 44, color = "black"),
    axis.line.x = element_line(linewidth = 0.6, color = "black"),
    legend.position = "none",
    panel.spacing.x = unit(0.5, "lines"),
    panel.grid.major.y = element_line(color = "gray95", linewidth = 0.2, linetype = "dashed"),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_line(color = "gray95", linewidth = 0.2, linetype = "dashed"),
    panel.grid.minor.x = element_blank(),
    panel.border = element_blank(),
    axis.ticks.x = element_line(linewidth = 0.6),
    axis.ticks.y = element_line(linewidth = 0.6),
    axis.ticks.length.y = unit(0, "cm"),
    axis.ticks.length.x = unit(.3, "cm"),
    plot.margin = margin(t = 12, r = 12, b = 12, l = 12)
  )

# Gain importance
importance_matrix <- xgb.importance(model = model_xgboost)
selected_features <- c("height", "density", "per_volume", "agr_height", "agr_density")
feature_labels <- c(
  height = "Building height",
  density = "Building density",
  per_volume = "Building volume per capita",
  agr_height = "Annual growth rate of building height",
  agr_density = "Annual growth rate of building density"
)
importance_matrix_selected <- importance_matrix[importance_matrix$Feature %in% selected_features, ]

p_gain <- ggplot(importance_matrix_selected, aes(x = reorder(Feature, Gain), y = Gain, fill = Gain)) +
  scale_y_continuous(limits = c(0, 0.11), breaks = seq(0, 0.09, 0.03), expand = c(0, 0)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = sprintf("%.2f", Gain)), hjust = -0.2, size = 15, color = "black") +
  scale_fill_gradient2(low = "#F1B6DA", high = "#C51B7D", midpoint = 0) +
  scale_x_discrete(labels = feature_labels) + 
  custom_theme +
  theme(
    axis.line.y = element_line(linewidth = 0.6, color = "black"),
    axis.text.y = element_text(margin = margin(r = 6)),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.line.x = element_blank(),
  ) +
  coord_flip() +
  labs(title = "", x = NULL, y = NULL)
p_gain
ggsave("healthyaging-mobileuser2-gain.png", plot = p_gain, width = 6.5, height = 3.5)

# SHAP values
shap_xgboost <- shapviz(model_xgboost, X_pred = as.matrix(traindata[, 2:(ncol(traindata)-1)]))
shap_xgboost$S <- shap_xgboost$S[, selected_features, drop = FALSE]
shap_values <- shap_xgboost$S
shap_summary <- data.frame(
  variable = colnames(shap_values),
  mean_shap = apply(abs(shap_values), 2, mean)
)

p_shap_bar <- ggplot(shap_summary, aes(x = reorder(variable, mean_shap), y = mean_shap, fill = mean_shap)) +
  scale_y_continuous(limits = c(0, 0.15), breaks = seq(0, 0.09, 0.03), expand = c(0, 0)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = sprintf("%.2f", mean_shap)), hjust = -0.2, size = 15, color = "black") +
  scale_fill_gradient2(mid = "#F7F7F7", high = "#C51B7D", midpoint = 0) +
  scale_x_discrete(labels = feature_labels) + 
  custom_theme +
  theme(
    axis.line.y = element_line(linewidth = 0.6, color = "black"),
    axis.text.y = element_text(margin = margin(r = 6)),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.line.x = element_blank(),
  ) +
  coord_flip() +
  labs(title = "", x = NULL, y = NULL)
p_shap_bar
ggsave("healthyaging-mobileuser2-shap.png", plot = p_shap_bar, width = 6.5, height = 3.5)

# Beeswarm plot
p_beeswarm <- sv_importance(shap_xgboost, kind = "beeswarm") +
  scale_color_gradientn(
    colours = c("#376A18","#4D9221", "#B8E186", "#F7F7F7", "#F1B6DA", "#C51B7D","#94145D"),
    name = "Feature value"
  ) +
  scale_x_continuous(
    limits = c(-0.6, 0.6),
    breaks = seq(-0.6, 0.6, 0.3),
    labels = function(x) ifelse(x == 0, "0", format(x, nsmall = 1)),
    expand = c(0, 0)
  ) +
  custom_theme +
  theme(
    axis.text.y = element_blank(),
    plot.margin = margin(t = 10, r = 20, b = 10, l = 20),
    axis.text.x = element_text(size = 44, color = "black", margin = margin(t = 6))
  ) +
  labs(title = "", x = NULL, y = NULL)
p_beeswarm
ggsave("healthyaging-mobileuser2-bee.png", plot = p_beeswarm, width = 3.5, height = 3.7)