# Hier wollen wir ja immer einen linear Zusammenhang darstellen (CTQ ~ GAF) und dann die Residuen berechnen.
# 1) mittels PCA: einmal PCA mit allen Indikatoren für e.g. Env. Risk und eine PCA mit allen Indikatoren für Functioning.
# Dann jeweils die erste Komponente mit einander korrelieren und die Residuen ausgeben.
# Für 1) input: matrix 1 & matrix 2, output: residuen

set.seed(1)
test_1 <- matrix(rnorm(1500), nrow = 150)
test_2 <- matrix(rnorm(1500), nrow = 150)

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

GetResidualsPCA(test_1, test_2)

# Für 2) input: task & mlr learner, output: residuen
# 2) Mittels ML: ein ML Model aufstellen in welchem durch Prädiktoren (z.B. Env. Risk Faktoren) eine Variable (Functioning) prädiziert wird, dann die Residuen ausgeben.
