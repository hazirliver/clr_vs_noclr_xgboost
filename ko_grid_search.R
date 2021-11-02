library(devtools)
library(tidyverse)
library(xgboost)
library(caret)
library(pROC)

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
                           print_every_n = 50,
                           min_child_weight = current_MinChild,
                           gamma = current_Gamma,
                           booster = "gbtree",
                           early_stopping_rounds = 70,
                           nthread = 6)


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


fit_predict <- function(df, param)
{
  # Обучение модели ----
  # Рассплитим данные в трейн и тест
  train <- sample_frac(df, 0.75)
  test <- setdiff(df, train)

  # Подготовим данные в нужном XgBoost`у формате
  xgb.train.data <- xgb.DMatrix(data.matrix(train[2:(ncol(train) - 1)]),
                                label = train$labels_Response, missing = NA)

  xgb.test.data <- xgb.DMatrix(data.matrix(test[2:(ncol(test) - 1)]),
                               label = test$labels_Response, missing = NA)


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
              train = train,
              test = test,
              xgb.train.data = xgb.train.data,
              xgb.test.data = xgb.test.data
  ))
}

predict_error <- function(df, param.list)
{
  # Обучение модели
  model_list <- fit_predict(df, param.list)

  # Предсказание
  pred <- predict(model_list$model, model_list$xgb.test.data)

  predicted_values <- data.frame(true_labels = model_list$test$Response,
                                 predicted_values = if_else(pred < 0.5, "NR", "R"))


  # Анализ предсказания ----
  # Считаем ошибку предсказания (error rate)
  error_rate <- sum(predicted_values$true_labels != predicted_values$predicted_values) / nrow(predicted_values)
  error_rate

  # Считаем AUC
  xgb.roc_obj <- roc(model_list$test$labels_Response, pred)
  auc <- auc(xgb.roc_obj)

  return(c(error_rate, auc))
}








# Препроцессинг данных ----
# Фиксируем сид для воспроизводимости результата
set.seed(1337)

# Испортируем данные
ko <- read_tsv("./input_data/ko", col_types = "c")
#mpa_org <- read_tsv("./input_data/mpa_org", col_types = "c")
metadata <- read_tsv("./input_data/combined_metadata",
                     col_types = cols(.default = "f", sampleid = "c"))


# Смерджим ko и metadata
ko_meta <- inner_join(metadata, ko, "sampleid")


# Отберем только больных, чтобы разделять респондеров от нереспондеров
ko_meta_nh <- ko_meta %>%
  dplyr::filter(Disease != "Healthy") %>%
  mutate(labels_Response = if_else(Response == "R", 1, 0))

ko_meta_nh <- ko_meta_nh %>%
  dplyr::select_at(c(2,5:ncol(ko_meta_nh)))

ko_meta_nh_train <- sample_frac(ko_meta_nh, 0.75)

xgb.train.data <- xgb.DMatrix(data.matrix(ko_meta_nh_train[2:(ncol(ko_meta_nh_train) - 1)]),
                              label = ko_meta_nh_train$labels_Response, missing = NA)


searchGridSubCol <- expand.grid(max_depth = 2:8,
                                eta = c(1, 0.1, 0.01),
                                gamma = 1:3,
                                min_child = 1:3)


griding_fit_predict <- apply(searchGridSubCol, 1,
                             \(x) grid_search(dtrain = xgb.train.data,
                                              current_depth = x[1],
                                              current_eta = x[2],
                                              current_Gamma = x[3],
                                              current_MinChild = x[4]))


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


multiple_test <- t(replicate(20, predict_error(ko_meta_nh, param.list)))
colnames(multiple_test) <- c("error_rate", "auc")
test_with_all_variables <- apply(multiple_test, 2, mean)





##############################################################################
# Removing highly correlated features ----------------------------------------


tmp <- cor(ko[2:ncol(ko)])
tmp[upper.tri(tmp)] <- 0
diag(tmp) <- 0

ko0 <- ko[2:ncol(ko)]
ko_new <- ko0[, !apply(tmp, 2, function(x) any(abs(x) > 0.85, na.rm = TRUE))]
ko_new$sampleid <- ko$sampleid


ko_meta_new <- inner_join(metadata, ko_new, "sampleid")


# Отберем только больных, чтобы разделять респондеров от нереспондеров
ko_meta_nh_new <- ko_meta_new %>%
  dplyr::filter(Disease != "Healthy") %>%
  mutate(labels_Response = if_else(Response == "R", 1, 0))

ko_meta_nh_new <- ko_meta_nh_new %>%
  dplyr::select_at(c(2,5:ncol(ko_meta_nh_new)))

ko_meta_nh_train_new <- sample_frac(ko_meta_nh_new, 0.75)

xgb.train.data_new <- xgb.DMatrix(data.matrix(ko_meta_nh_train_new[2:(ncol(ko_meta_nh_train_new) - 1)]),
                              label = ko_meta_nh_train_new$labels_Response, missing = NA)


searchGridSubCol <- expand.grid(max_depth = 2:8,
                                eta = c(1, 0.1, 0.01),
                                gamma = 1:3,
                                min_child = 1:3)


griding_fit_predict_new <- apply(searchGridSubCol, 1,
                             \(x) grid_search(dtrain = xgb.train.data_new,
                                              current_depth = x[1],
                                              current_eta = x[2],
                                              current_Gamma = x[3],
                                              current_MinChild = x[4]))



output_new <- t(griding_fit_predict_new)
colnames(output_new) <- c("Train error",
                      "Test error",
                      "max_depth",
                      "eta",
                      "gamma",
                      "min_child_weight")

output_new <- as_tibble(output_new)


output_new %>%
  arrange(`Test error`) %>%
  head()

param <- output[output$`Test error` == min(output$`Test error`),][3:ncol(output)]
param.list <- Map(cbind, split.default(param[-1], names(param)[-1]))


# Предсказание
multiple_test_new <- t(replicate(20, predict_error(ko_meta_nh_new, param.list)))
colnames(multiple_test_new) <- c("error_rate", "auc")
test_with_0.8 <- apply(multiple_test_new, 2, mean)



# TODO Протестить с <0.99
