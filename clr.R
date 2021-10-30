library(devtools)
library(tidyverse)
library(xgboost)
library(caret)
library(pROC)
library(compositions)

install_github("AppliedDataSciencePartners/xgboostExplainer")
library(xgboostExplainer)

set.seed(1337)

# Испортируем данные
ko <- read_tsv("./input_data/ko", col_types = "c")
mpa_org <- read_tsv("./input_data/mpa_org", col_types = "c")
metadata <- read_tsv("./input_data/combined_metadata",
                     col_types = cols(.default = "f", sampleid = "c"))


# Применим clr к данным
mpa_org_clr <- cbind(mpa_org$sampleid,
                     as.data.frame(matrix(as.numeric(clr(mpa_org[2:ncol(mpa_org)])),
                                          ncol = ncol(mpa_org) -1)))
colnames(mpa_org_clr) <- colnames(mpa_org)


# Смерджим mpa_org и metadata
mpa_meta_clr <- inner_join(metadata, mpa_org_clr, "sampleid")


# Отберем только больных, чтобы разделять респондеров от нереспондеров
mpa_meta_nh_clr <- mpa_meta_clr %>%
  dplyr::filter(Disease != "Healthy") %>%
  mutate(labels_Response = if_else(Response == "R", 1, 0))

mpa_meta_nh_clr <- mpa_meta_nh_clr %>%
  dplyr::select_at(c(2,5:ncol(mpa_meta_nh_clr)))

# Обучение модели ----
# Рассплитим данные в трейн и тест
mpa_meta_nh_clr_train <- sample_frac(mpa_meta_nh_clr, 0.75)
mpa_meta_nh_clr_test <- setdiff(mpa_meta_nh_clr, mpa_meta_nh_clr_train)


# Создадим фолды для кроссвалидации и подбора
cv_clr <- createFolds(mpa_meta_nh_clr_train[2:ncol(mpa_meta_nh_clr_train)], k = 10)
ctrl_clr <- trainControl(method = "cv",index = cv_clr)

# Подготовим данные в нужном XgBoost`у формате
xgb.train.data_clr <- xgb.DMatrix(data.matrix(mpa_meta_nh_clr_train[2:(ncol(mpa_meta_nh_clr_train) - 1)]),
                                  label = mpa_meta_nh_clr_train$labels_Response, missing = NA)

xgb.test.data_clr <- xgb.DMatrix(data.matrix(mpa_meta_nh_clr_test[2:(ncol(mpa_meta_nh_clr_train) - 1)]),
                                 label = mpa_meta_nh_clr_test$labels_Response, missing = NA)

# Поймем оптимальное количество итераций для обучения модели
param_clr <- list(objective = "binary:logistic", base_score = 0.5)
xgboost.cv_clr <- xgb.cv(param=param_clr, data = xgb.train.data_clr,
                         folds = cv_clr, nrounds = 1500, early_stopping_rounds = 100,
                         metrics='error', nthread = 8)
best_iteration_clr <- xgboost.cv_clr$best_iteration

# Обучение модели
model2_clr <- xgboost(data = as.matrix(mpa_meta_nh_clr_train[2:(ncol(mpa_meta_nh_clr_train) - 1)]),
                      label = mpa_meta_nh_clr_train$labels_Response, eta = 1,
                      nthread = 8, nrounds = best_iteration_clr, objective = "binary:logistic")

# Предсказание
pred2_clr <- predict(model2_clr, as.matrix(mpa_meta_nh_clr_test[2:(ncol(mpa_meta_nh_clr_test) - 1)]))

predicted_values_clr <- data.frame(true_labels = mpa_meta_nh_clr_test$Response,
                                   predicted_values = if_else(pred2_clr < 0.5, "NR", "R"))

# Анализ предсказания ----
# Считаем ошибку предсказания (Accuracy)
sum(predicted_values_clr$true_labels != predicted_values_clr$predicted_values) / nrow(predicted_values_clr)

# Считаем AUC
xgb.roc_obj_clr <- roc(mpa_meta_nh_clr_test$labels_Response, pred2_clr)
auc(xgb.roc_obj_clr)

# Построим диаграмму важности каждой переменной (бактерии) в совокупности
col_names_clr <-  attr(xgb.train.data_clr, ".Dimnames")[[2]]
imp_clr <- xgb.importance(col_names_clr, model2_clr)
xgb.plot.importance(imp_clr)

?xgb.importance
head(imp_clr)


# Посмотрим на индивидуальнй вклад каждого признака в какой-нибудь семпл
explainer_clr <- buildExplainer(model2_clr,
                                xgb.train.data_clr,
                                type="binary",
                                base_score = 0.5,
                                trees_idx = NULL)
pred.breakdown_clr <- explainPredictions(model2_clr, explainer_clr, xgb.test.data_clr)

# Например, в семпл с номером 55
idx_to_get_clr <- as.integer(55)
mpa_meta_nh_clr_test[idx_to_get_clr, 2:(ncol(mpa_meta_nh_clr_test) - 1)]

watefall_plot_clr <- showWaterfall(model2_clr, explainer_clr, xgb.test.data_clr,
                                   data.matrix(mpa_meta_nh_clr_test[2:(ncol(mpa_meta_nh_clr_test) - 1)]) ,
                                   idx_to_get_clr, type = "binary")
watefall_plot_clr
