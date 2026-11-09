
# Q23 - Load Titanic training data

library(readr)

train <- read_csv("data/train.csv", show_col_types = FALSE)

print("Training data dimensions:")
print(dim(train))

print("First 6 rows:")
print(head(train))


# Q24 - Repeat Titanic analysis in R

# Q14 - Explore the data
cat("\n===== Data Exploration =====\n")
print(colSums(is.na(train)))
print(table(train$Survived))
print(tapply(train$Survived, train$Sex, mean))

# Clean missing values using training statistics
age_median <- median(train$Age, na.rm = TRUE)
fare_median <- median(train$Fare, na.rm = TRUE)
embarked_mode <- names(sort(table(train$Embarked), decreasing = TRUE))[1]

prepare_data <- function(df) {
  df$Age[is.na(df$Age)] <- age_median
  df$Fare[is.na(df$Fare)] <- fare_median
  df$Embarked[is.na(df$Embarked)] <- embarked_mode

  data.frame(
    Pclass = df$Pclass,
    Age = df$Age,
    SibSp = df$SibSp,
    Parch = df$Parch,
    Fare = df$Fare,
    Sex_female = as.integer(df$Sex == "female"),
    Embarked_C = as.integer(df$Embarked == "C"),
    Embarked_Q = as.integer(df$Embarked == "Q")
  )
}

train_clean <- prepare_data(train)
train_clean$Survived <- train$Survived

cat("\n===== Cleaned Data =====\n")
print(head(train_clean))
print(colSums(is.na(train_clean)))

# Q15 - Logistic Regression
model <- glm(
  Survived ~ .,
  data = train_clean,
  family = binomial()
)

cat("\nLogistic Regression trained successfully!\n")

# Q16 - Training Accuracy
train_prob <- predict(model, train_clean, type = "response")
train_pred <- ifelse(train_prob >= 0.5, 1, 0)
train_accuracy <- mean(train_pred == train_clean$Survived)

cat("\nTraining Accuracy:", round(train_accuracy, 4), "\n")

# Q17 - Predict Kaggle test.csv
test <- read_csv("data/test.csv", show_col_types = FALSE)
test_clean <- prepare_data(test)

test_prob <- predict(model, test_clean, type = "response")
test_pred <- ifelse(test_prob >= 0.5, 1, 0)

cat("\nTest data dimensions:\n")
print(dim(test))
cat("First 10 predictions:\n")
print(head(test_pred, 10))
cat("Total predictions:", length(test_pred), "\n")

# Q18 - Holdout Evaluation
# Kaggle test.csv does not contain true labels

set.seed(42)
train_index <- sample(
  seq_len(nrow(train_clean)),
  size = floor(0.8 * nrow(train_clean))
)

holdout_train <- train_clean[train_index, ]
holdout_test <- train_clean[-train_index, ]

holdout_model <- glm(
  Survived ~ .,
  data = holdout_train,
  family = binomial()
)

holdout_prob <- predict(
  holdout_model,
  holdout_test,
  type = "response"
)

holdout_pred <- ifelse(holdout_prob >= 0.5, 1, 0)

holdout_accuracy <- mean(
  holdout_pred == holdout_test$Survived
)

cat("\nHoldout Test Accuracy:",
    round(holdout_accuracy, 4), "\n")

# Q19 - Print Summary
cat("\n===== Titanic R Analysis Summary =====\n")
cat("Training observations:", nrow(train), "\n")
cat("Kaggle test observations:", nrow(test), "\n")
cat("Training Accuracy:", round(train_accuracy, 4), "\n")
cat("Holdout Test Accuracy:", round(holdout_accuracy, 4), "\n")
cat("Kaggle test predictions:", length(test_pred), "\n")
cat("====================================\n")
