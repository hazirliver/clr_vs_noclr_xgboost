library(devtools)
library(tidyverse)
library(xgboost)
library(caret)
library(pROC)
library(compositions)

#install_github("AppliedDataSciencePartners/xgboostExplainer")
library(xgboostExplainer)


grid_search <- function(dtrain,
                        current_depth,
                        current_eta,
                        current_Gamma,
                        current_MinChild){

  xgboostModelCV <- xgb.cv(data =  dtrain,
                           nrounds = 1000,
                           nfold = 7,
                           metrics = "error",
                           objective = "binary:logistic",
                           max.depth = current_depth,
                           eta = current_eta,
                           print_every_n = 20,
                           min_child_weight = current_MinChild,
                           gamma = current_Gamma,
                           booster = "gbtree",
                           early_stopping_rounds = 70)


  xvalidationScores <- as.data.frame(xgboostModelCV$evaluation_log)

  train_error <- tail(xvalidationScores$train_error_mean, 1)
  test_error <- tail(xvalidationScores$test_error_mean, 1)

  output <- return(c(train_error,
                     test_error,
                     current_depth,
                     current_eta,
                     current_Gamma,
                     current_MinChild))
}


fit_predict <- function(mpa_meta_nh, param)
{
  # Обучение модели ----
  # Рассплитим данные в трейн и тест
  mpa_meta_nh_train <- sample_frac(mpa_meta_nh, 0.75)
  mpa_meta_nh_test <- setdiff(mpa_meta_nh, mpa_meta_nh_train)

  # Подготовим данные в нужном XgBoost`у формате
  xgb.train.data <- xgb.DMatrix(data.matrix(mpa_meta_nh_train[2:(ncol(mpa_meta_nh_train) - 1)]),
                                label = mpa_meta_nh_train$labels_Response, missing = NA)

  xgb.test.data <- xgb.DMatrix(data.matrix(mpa_meta_nh_test[2:(ncol(mpa_meta_nh_test) - 1)]),
                               label = mpa_meta_nh_test$labels_Response, missing = NA)


  # Поймем оптимальное количество итераций для обучения модели
  param_cur <- param

  best_iteration <- 150

  # Обучение модели
  model <- xgboost(data = xgb.train.data,
                   params = param_cur,
                   nrounds = best_iteration,
                   objective = "binary:logistic",
                   metrics='error',
                   print_every_n = 20)

  return(list(model = model,
              mpa_meta_nh_train = mpa_meta_nh_train,
              mpa_meta_nh_test = mpa_meta_nh_test,
              xgb.train.data = xgb.train.data,
              xgb.test.data = xgb.test.data
  ))
}

predict_error <- function(mpa_meta_nh, param.list)
{
  # Обучение модели
  model_list <- fit_predict(mpa_meta_nh, param.list)

  # Предсказание
  pred <- predict(model_list$model, model_list$xgb.test.data)

  predicted_values <- data.frame(true_labels = model_list$mpa_meta_nh_test$Response,
                                 predicted_values = if_else(pred < 0.5, "NR", "R"))


  # Анализ предсказания ----
  # Считаем ошибку предсказания (error rate)
  error_rate <- sum(predicted_values$true_labels != predicted_values$predicted_values) / nrow(predicted_values)
  error_rate

  # Считаем AUC
  xgb.roc_obj <- roc(model_list$mpa_meta_nh_test$labels_Response, pred)
  auc <- auc(xgb.roc_obj)

  return(c(error_rate, auc))
}







# Препроцессинг данных ----
# Фиксируем сид для воспроизводимости результата
set.seed(1337)

# Испортируем данные
#ko <- read_tsv("./input_data/ko", col_types = "c")
mpa_org <- read_tsv("./input_data/mpa_org", col_types = "c")
metadata <- read_tsv("./input_data/combined_metadata",
                     col_types = cols(.default = "f", sampleid = "c"))

# Применим clr к данным
mpa_org_clr <- cbind(mpa_org$sampleid,
                     as.data.frame(matrix(as.numeric(clr(mpa_org[2:ncol(mpa_org)])),
                                          ncol = ncol(mpa_org) -1)))
colnames(mpa_org_clr) <- colnames(mpa_org)

# Смерджим mpa_org и metadata
mpa_meta <- inner_join(metadata, mpa_org, "sampleid")

# Отберем только больных, чтобы разделять респондеров от нереспондеров
mpa_meta_nh <- mpa_meta %>%
  dplyr::filter(Disease != "Healthy") %>%
  mutate(labels_Response = if_else(Response == "R", 1, 0))

mpa_meta_nh <- mpa_meta_nh %>%
  dplyr::select_at(c(2,5:ncol(mpa_meta_nh)))

mpa_meta_nh_train <- sample_frac(mpa_meta_nh, 0.75)

xgb.train.data <- xgb.DMatrix(data.matrix(mpa_meta_nh_train[2:(ncol(mpa_meta_nh_train) - 1)]),
                              label = mpa_meta_nh_train$labels_Response, missing = NA)


searchGridSubCol <- expand.grid(max_depth = 2:8,
                                eta = c(1, 0.1, 0.01),
                                gamma = 1:3,
                                min_child = 1:3)


system.time(
  griding_fit_predict <- apply(searchGridSubCol, 1,
                               \(x) grid_search(dtrain = xgb.train.data,
                                                current_depth = x[1],
                                                current_eta = x[2],
                                                current_Gamma = x[3],
                                                current_MinChild = x[4]))
)

output <- t(griding_fit_predict)
colnames(output) <- c("Train error",
                      "Test error",
                      "max_depth",
                      "eta",
                      "gamma",
                      "min_child_weight")

output <- as_tibble(output)


output %>%
  arrange(`Test error`) %>%
  head()

param <- output[output$`Test error` == min(output$`Test error`),][3:ncol(output)]
param.list <- Map(cbind, split.default(param[-1], names(param)[-1]))

# Обучение модели
model_list <- fit_predict(mpa_meta_nh, param.list)

# Предсказание
multiple_test <- t(replicate(20, predict_error(mpa_meta_nh, param.list)))
colnames(multiple_test) <- c("error_rate", "auc")
apply(multiple_test, 2, mean)

pred <- predict(model_list$model, model_list$xgb.test.data)

predicted_values <- data.frame(true_labels = model_list$mpa_meta_nh_test$Response,
                               predicted_values = if_else(pred < 0.5, "NR", "R"))


# Анализ предсказания ----
# Считаем ошибку предсказания (error rate)
error_rate <- sum(predicted_values$true_labels != predicted_values$predicted_values) / nrow(predicted_values)
error_rate

# Считаем AUC
xgb.roc_obj <- roc(model_list$mpa_meta_nh_test$labels_Response, pred)
auc(xgb.roc_obj)


# Построим диаграмму важности каждой переменной (бактерии) в совокупности
col_names <-  attr(model_list$xgb.train.data, ".Dimnames")[[2]]
imp <- xgb.importance(col_names, model_list$model)
xgb.plot.importance(imp)

?xgb.importance
head(imp)


# Посмотрим на индивидуальнй вклад каждого признака в какой-нибудь семпл
explainer <- buildExplainer(model_list$model,
                            model_list$xgb.train.data,
                            type="binary",
                            base_score = 0.5,
                            trees_idx = NULL)
pred.breakdown <- explainPredictions(model_list$model, explainer, model_list$xgb.test.data)

# Например, в семпл с номером 55
idx_to_get <- as.integer(55)
model_list$mpa_meta_nh_test[idx_to_get, 2:(ncol(model_list$mpa_meta_nh_test) - 1)]

watefall_plot <- showWaterfall(model_list$model, explainer, model_list$xgb.test.data,
                               data.matrix(model_list$mpa_meta_nh_test[2:(ncol(model_list$mpa_meta_nh_test) - 1)]) ,
                               idx_to_get, type = "binary")
watefall_plot



# Например, в семпл с номером 26
idx_to_get <- as.integer(26)
model_list$mpa_meta_nh_test[idx_to_get, 2:(ncol(model_list$mpa_meta_nh_test) - 1)]

watefall_plot <- showWaterfall(model_list$model, explainer, model_list$xgb.test.data,
                               data.matrix(model_list$mpa_meta_nh_test[2:(ncol(model_list$mpa_meta_nh_test) - 1)]) ,
                               idx_to_get, type = "binary")
watefall_plot

