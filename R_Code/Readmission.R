# ---- Packages ----
pkgs <- c(
  "boot","leaps","MASS","glmnet","caret","carData","car","corrplot","Hmisc",
  "ggplot2","dplyr","pROC","randomForest","RColorBrewer"
)
to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))

# ---- Data ----
fullData <- read.csv("hospital_readmissions.csv", header = TRUE, stringsAsFactors = FALSE)

# Ensure canonical types early
fullData <- fullData |>
  mutate(
    across(c(age, medical_specialty, diag_1, diag_2, diag_3, glucose_test, A1Ctest), as.factor),
    readmitted = factor(readmitted) # we'll normalize levels below
  )

# Normalize binary outcome levels to c("no","yes") for consistency
if (!all(levels(fullData$readmitted) %in% c("no","yes"))) {
  # guess: first level = "no", others = "yes"
  fullData$readmitted <- factor(ifelse(fullData$readmitted %in% c("yes","1","Y","readmitted"), "yes", "no"),
                                levels = c("no","yes"))
}

set.seed(100)
test_idx  <- sample(nrow(fullData), size = floor(0.2*nrow(fullData)))
testData  <- fullData[test_idx, ]
trainData <- fullData[-test_idx, ]

# ---- EDA (examples) ----
blues9 <- RColorBrewer::brewer.pal(9, "Blues")

count_df <- trainData |>
  group_by(age, readmitted) |>
  summarise(count = n(), .groups = "drop")

ggplot(count_df, aes(x = age, y = count, fill = readmitted)) +
  geom_col() + labs(x = "Age", y = "Count", fill = "Readmitted") +
  theme_minimal()

boxplot(time_in_hospital ~ readmitted, data = trainData,
        xlab = "Readmitted", ylab = "time_in_hospital", col = blues9)

# ---- Full logistic model ----
full_formula <- readmitted ~ .
model_full <- glm(full_formula, data = trainData, family = binomial())
summary(model_full)

# Mallows' Cp for GLM (for reference)
RSS    <- sum(residuals(model_full, type = "deviance")^2)
n      <- nrow(trainData)
p      <- length(coef(model_full))
sigma2 <- RSS / (n - p)
Cp     <- (RSS / sigma2) - n + 2*p

# ---- Stepwise (BIC via stepAIC with k = log(n)) ----
model_step <- MASS::stepAIC(model_full, direction = "backward", k = log(n), trace = FALSE)
summary(model_step)

# ---- Glmnet design matrices (freeze terms on training) ----
# Use the *training* terms so train/test columns are consistent
# Build terms on TRAINING data and drop the response
terms_x <- delete.response(terms(full_formula, data = trainData))

# Lock factor levels using model.frame; this prevents new/absent levels chaos
mf_train <- model.frame(terms_x, data = trainData, na.action = na.pass)
mf_test  <- model.frame(
  terms_x, data = testData, na.action = na.pass,
  xlev = lapply(mf_train, function(v) if (is.factor(v)) levels(v) else NULL)
)
X_train <- model.matrix(terms_x, data = trainData)[, -1, drop = FALSE]
X_test  <- model.matrix(terms_x, data = testData)[,  -1, drop = FALSE]

# Response 0/1 for glmnet
y_train <- as.integer(trainData$readmitted == "yes")
y_test  <- as.integer(testData$readmitted == "yes")

# ---- Ridge / Elastic Net / Lasso ----
cv_ridge <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 0, nfolds = 10)
fit_ridge <- glmnet(X_train, y_train, family = "binomial", alpha = 0, lambda = cv_ridge$lambda.min)

cv_en <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 0.5, nfolds = 10)
fit_en <- glmnet(X_train, y_train, family = "binomial", alpha = 0.5, lambda = cv_en$lambda.min)

cv_lasso <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 1, nfolds = 10)
fit_lasso <- glmnet(X_train, y_train, family = "binomial", alpha = 1, lambda = cv_lasso$lambda.min)

# ---- Predictions + ROC/AUC helpers ----
pred_bin <- function(prob, thr = 0.5) as.integer(prob >= thr)

eval_bin <- function(y_true01, y_hat01) {
  # caret::confusionMatrix wants factors with identical level order
  cm <- caret::confusionMatrix(
    factor(y_hat01, levels = c(0,1), labels = c("no","yes")),
    factor(y_true01, levels = c(0,1), labels = c("no","yes"))
  )
  list(
    cm = cm$table,
    Accuracy = unname(cm$overall["Accuracy"]),
    Precision = unname(cm$byClass["Pos Pred Value"]),
    Recall = unname(cm$byClass["Sensitivity"]),
    F1 = unname(cm$byClass["F1"])
  )
}

plot_roc <- function(y_true01, score, main = "ROC") {
  ro <- pROC::roc(response = y_true01, predictor = as.numeric(score))
  plot(ro, main = sprintf("%s (AUC = %.3f)", main, pROC::auc(ro)))
  invisible(ro)
}

# LASSO
prob_lasso <- as.numeric(predict(fit_lasso, X_test, type = "response"))
yhat_lasso <- pred_bin(prob_lasso)
plot_roc(y_test, prob_lasso, "LASSO ROC")
metrics_lasso <- eval_bin(y_test, yhat_lasso)

# Ridge
prob_ridge <- as.numeric(predict(fit_ridge, X_test, type = "response"))
yhat_ridge <- pred_bin(prob_ridge)
plot_roc(y_test, prob_ridge, "Ridge ROC")
metrics_ridge <- eval_bin(y_test, yhat_ridge)

# Elastic Net
prob_en <- as.numeric(predict(fit_en, X_test, type = "response"))
yhat_en <- pred_bin(prob_en)
plot_roc(y_test, prob_en, "Elastic Net ROC")
metrics_en <- eval_bin(y_test, yhat_en)

# ---- Full and Step models evaluated on test ----
prob_full <- predict(model_full, newdata = testData, type = "response")
yhat_full <- pred_bin(prob_full)
plot_roc(y_test, prob_full, "Full GLM ROC")
metrics_full <- eval_bin(y_test, yhat_full)

prob_step <- predict(model_step, newdata = testData, type = "response")
yhat_step <- pred_bin(prob_step)
plot_roc(y_test, prob_step, "Stepwise (BIC) ROC")
metrics_step <- eval_bin(y_test, yhat_step)

# ---- Parsimonious GLM (your model3) ----
model3 <- glm(readmitted ~ n_outpatient + n_inpatient + n_emergency + diabetes_med,
              data = trainData, family = binomial())
prob_m3 <- predict(model3, newdata = testData, type = "response")
yhat_m3 <- pred_bin(prob_m3)
plot_roc(y_test, prob_m3, "Model3 ROC")
metrics_m3 <- eval_bin(y_test, yhat_m3)

# ---- Random Forest ----
rf_model <- randomForest(readmitted ~ n_outpatient + n_inpatient + n_emergency + diabetes_med,
                         data = trainData, ntree = 500)
pred_rf_class <- predict(rf_model, newdata = testData, type = "response") # factor "no"/"yes"
yhat_rf <- as.integer(pred_rf_class == "yes")
# For ROC, we prefer probabilities; use type="prob"
prob_rf <- predict(rf_model, newdata = testData, type = "prob")[, "yes"]
plot_roc(y_test, prob_rf, "Random Forest ROC")
metrics_rf <- eval_bin(y_test, yhat_rf)

# ---- Simple correlation visuals (numeric subset) ----
num_cols <- sapply(fullData, is.numeric)
df_num <- fullData[, num_cols, drop = FALSE]
if (ncol(df_num) >= 2) {
  data.cor <- cor(df_num, method = "spearman", use = "pairwise.complete.obs")
  palette <- colorRampPalette(brewer.pal(10, "RdYlBu"))(256)
  heatmap(x = data.cor, col = palette, symm = TRUE)
  corrplot::corrplot(data.cor, type = "upper", order = "hclust",
                     tl.cex = 0.7, method = "color", addCoef.col = "black")
}

# ---- Compare model metrics (printed) ----
list(
  LASSO = metrics_lasso,
  Ridge = metrics_ridge,
  ElasticNet = metrics_en,
  FullGLM = metrics_full,
  Step_BIC = metrics_step,
  GLM_small = metrics_m3,
  RandomForest = metrics_rf
)
