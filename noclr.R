library(devtools)
library(tidyverse)
library(xgboost)
library(caret)
library(pROC)

install_github("AppliedDataSciencePartners/xgboostExplainer")
library(xgboostExplainer)


# Препроцессинг данных ----
# Фиксируем сид для воспроизводимости результата
set.seed(1337)

# Испортируем данные
ko <- read_tsv("./input_data/ko", col_types = "c")
mpa_org <- read_tsv("./input_data/mpa_org", col_types = "c")
metadata <- read_tsv("./input_data/combined_metadata",
                     col_types = cols(.default = "f", sampleid = "c"))

# Смерджим mpa_org и metadata
mpa_meta <- inner_join(metadata, mpa_org, "sampleid")

# Отберем только больных, чтобы разделять респондеров от нереспондеров
mpa_meta_nh <- mpa_meta %>%
  dplyr::filter(Disease != "Healthy") %>%
  mutate(labels_Response = if_else(Response == "R", 1, 0))

mpa_meta_nh <- mpa_meta_nh %>%
  dplyr::select_at(c(2,5:ncol(mpa_meta_nh)))


# Обучение модели ----
# Рассплитим данные в трейн и тест
mpa_meta_nh_train <- sample_frac(mpa_meta_nh, 0.75)
mpa_meta_nh_test <- setdiff(mpa_meta_nh, mpa_meta_nh_train)


# Создадим фолды для кроссвалидации и подбора
cv <- createFolds(mpa_meta_nh_train[2:ncol(mpa_meta_nh_train)], k = 10)
ctrl <- trainControl(method = "cv",index = cv)

# Подготовим данные в нужном XgBoost`у формате
xgb.train.data <- xgb.DMatrix(data.matrix(mpa_meta_nh_train[2:(ncol(mpa_meta_nh_train) - 1)]),
                              label = mpa_meta_nh_train$labels_Response, missing = NA)

xgb.test.data <- xgb.DMatrix(data.matrix(mpa_meta_nh_test[2:(ncol(mpa_meta_nh_train) - 1)]),
                             label = mpa_meta_nh_test$labels_Response, missing = NA)


# Поймем оптимальное количество итераций для обучения модели
param <- list(objective = "binary:logistic", base_score = 0.5)

xgboost.cv <- xgb.cv(param=param, data = xgb.train.data,
                     folds = cv, nrounds = 1500, early_stopping_rounds = 100,
                     metrics='error',nthread = 8)
best_iteration <- xgboost.cv$best_iteration


# Обучение модели
model <- xgboost(data = as.matrix(mpa_meta_nh_train[2:(ncol(mpa_meta_nh_train) - 1)]),
                  label = mpa_meta_nh_train$labels_Response, eta = 1,
                  nthread = 8, nrounds = best_iteration, objective = "binary:logistic")

# Предсказание
pred2 <- predict(model, as.matrix(mpa_meta_nh_test[2:(ncol(mpa_meta_nh_test) - 1)]))

predicted_values <- data.frame(true_labels = mpa_meta_nh_test$Response,
                               predicted_values = if_else(pred2 < 0.5, "NR", "R"))


# Анализ предсказания ----
# Считаем ошибку предсказания (Accuracy)
sum(predicted_values$true_labels != predicted_values$predicted_values) / nrow(predicted_values)

# Считаем AUC
xgb.roc_obj <- roc(mpa_meta_nh_test$labels_Response, pred2)
auc(xgb.roc_obj)


# Построим диаграмму важности каждой переменной (бактерии) в совокупности
col_names <-  attr(xgb.train.data, ".Dimnames")[[2]]
imp <- xgb.importance(col_names, model)
xgb.plot.importance(imp)

?xgb.importance
head(imp)


# Посмотрим на индивидуальнй вклад каждого признака в какой-нибудь семпл
explainer <- buildExplainer(model,
                            xgb.train.data,
                            type="binary",
                            base_score = 0.5,
                            trees_idx = NULL)
pred.breakdown <- explainPredictions(model, explainer, xgb.test.data)

# Например, в семпл с номером 55
idx_to_get <- as.integer(55)
mpa_meta_nh_test[idx_to_get, 2:(ncol(mpa_meta_nh_test) - 1)]

watefall_plot <- showWaterfall(model, explainer, xgb.test.data,
                               data.matrix(mpa_meta_nh_test[2:(ncol(mpa_meta_nh_test) - 1)]) ,
                               idx_to_get, type = "binary")
watefall_plot
