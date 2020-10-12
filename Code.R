# Hier wollen wir ja immer einen linear Zusammenhang darstellen (CTQ ~ GAF) und dann die Residuen berechnen.
# 1) mittels PCA: einmal PCA mit allen Indikatoren für e.g. Env. Risk und eine PCA mit allen Indikatoren für Functioning.
# Dann jeweils die erste Komponente mit einander korrelieren und die Residuen ausgeben.
# Für 1) input: matrix 1 & matrix 2, output: residuen


GetResidualsPCA <- function(input_risk, input_functioning) {
  # some initial checks
  if ((!is.data.frame(input_risk) &
       !is.matrix(input_risk)) |
      (!is.data.frame(input_functioning) &
       !is.matrix(input_functioning)))
    stop("Error: Input(s) are not in correct format (matrix or dataframe).")
  if (nrow(input_risk) != nrow(input_functioning))
    stop("Error: Different number of participants in input files. Please check.")
  if (nrow(input_risk) < ncol(input_risk))
    message("There are fewier observations than variables. Results might be unstable.")
  
  # compute PCA separately for risk and functioning variables
  pca_risk <-
    prcomp(input_risk, scale = TRUE)$x[, 1] # extract first component
  pca_functioning <-
    prcomp(input_functioning, scale = TRUE)$x[, 1] # extract first component
  
  # linear model: predict functioning by risk
  lm_resilience <- lm(pca_risk ~ pca_functioning)
  
  # extract & return residuals of this linear model
  return(data.frame(residuals = lm_resilience$residuals))
  
}

# some test data
set.seed(1)
test_1 <- matrix(rnorm(1500), nrow = 150)
test_2 <- matrix(rnorm(1500), nrow = 150)

GetResidualsPCA(test_1, test_2)
# works


library(dplyr)
# Für 2) input: task & mlr learner, output: residuen
# 2) Mittels ML: ein ML Model aufstellen in welchem durch Prädiktoren (z.B. Env. Risk Faktoren) eine Variable (Functioning) prädiziert wird, dann die Residuen ausgeben.

GetResidualsML <- function(task, learner, cv_folds, cv_reps, par_set, optimization_strategy) {
  # some initial checks
  if (learner$type != "regr")
    stop("Error: Learner type must be 'regression'.")
  if (learner$predict.type != "response")
    stop("Error: Predict type must be 'response'.")
  if (sum(task$task.desc$n.feat) > task$task.desc$size)
    warning("Task with more features than observations given. Results might be unstable")
  
  # define resampling scheme
  if (cv_folds <= 1) {
    warning("Number of folds must be at least 2 for CV. Setting cv_folds to 5.")
    cv_folds <- 5
  }
  if (cv_reps > 1) {
    rdesc <-
      mlr::makeResampleDesc("RepCV", folds = cv_folds, reps = cv_reps)
  } else {
    rdesc <- mlr::makeResampleDesc("CV", iters = cv_folds)
  }
  
  lrn <- mlr::makeTuneWrapper(learner, control = ctrl,
                        measures = list(mae), 
                        resampling = rdesc, 
                        par.set = par_set, 
                        show.info = FALSE)
  resampled_models <-
    mlr::resample(lrn, task = task, 
                  resampling = rdesc, 
                  extract = getTuneResult, 
                  show.info = FALSE)
  
  
  predictions <- getRRPredictions(resampled_models)$data
  
  residuals <-
    predictions %>% mutate(residuals = truth - response) %>%
    group_by(id) %>%
    summarize(residuals = mean(residuals))
  
  return(residuals)
  
}

# some test data
data(BostonHousing, package = "mlbench")
regr.lrn <-
  mlr::makeLearner("regr.gbm")
regr.task <-
  mlr::makeRegrTask(id = "bh", data = BostonHousing, target = "medv")
task <- regr.task
learner <- regr.lrn
ctrl <- mlr::makeTuneControlRandom(maxit=100)
ps <- makeParamSet(
  makeIntegerParam("n.trees", lower = 100, upper = 1000)
)

GetResidualsML(
  task = regr.task,
  learner = regr.lrn,
  cv_folds = 2,
  cv_reps = 0,
  par_set = ps,
  optimization_strategy = ctrl
)