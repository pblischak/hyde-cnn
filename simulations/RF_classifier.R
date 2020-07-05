library(tidyverse)
library(patchwork)
library(abcrf)
library(caret)

test_plot <- TRUE

for(cu in c("0.5", "1.0", "2.0")){
  no_hyb     <- read_csv(paste0("no_hybridization_", cu, "_tests.csv"))
  hyb_sp     <- read_csv(paste0("hybrid_speciation_", cu, "_tests.csv"))
  admix      <- read_csv(paste0("admixture_", cu, "_tests.csv"))
  admix_mig  <- read_csv(paste0("admixture_w_gflow_", cu, "_tests.csv"))
  
  cnn_truth <- factor(rep(c("no_hyb", "hyb_sp", "admix", "admix_mig"),
                   each = 10000))
  cnn_predicted <- factor(c(no_hyb$best_model,
                         hyb_sp$best_model,
                         admix$best_model,
                         admix_mig$best_model))
  cnn_cfm <- confusionMatrix(cnn_truth, cnn_predicted)
  print(cnn_cfm)
  
  print(paste0("** Processing ", cu, " CU simulations **"))
  train <- data.frame(model = rep(c("no_hyb", "hyb_sp", "admix", "admix_mig"),
                                  each = 7500),
                      D    = c(no_hyb$D[1:7500], hyb_sp$D[1:7500],
                               admix$D[1:7500], admix_mig$D[1:7500]),
                      fhom = c(no_hyb$f_hom[1:7500], hyb_sp$f_hom[1:7500],
                               admix$f_hom[1:7500], admix_mig$f_hom[1:7500]),
                      Dp   = c(no_hyb$D_p[1:7500], hyb_sp$D_p[1:7500],
                               admix$D_p[1:7500], admix_mig$D_p[1:7500]))
  
  # Test for bad values then randomize the order
  if(sum(is.na(train$D)) > 0){
    train <- train[!is.na(train$D),]
  }
  train <- train[sample(nrow(train)),]
  
  test <- data.frame(model = rep(c("no_hyb", "hyb_sp", "admix", "admix_mig"),
                                 each = 2500),
                     D  = c(no_hyb$D[7501:10000], hyb_sp$D[7501:10000],
                            admix$D[7501:10000], admix_mig$D[7501:10000]),
                     fhom = c(no_hyb$f_hom[7501:10000], hyb_sp$f_hom[7501:10000],
                            admix$f_hom[7501:10000], admix_mig$f_hom[7501:10000]),
                     Dp = c(no_hyb$D_p[7501:10000], hyb_sp$D_p[7501:10000],
                            admix$D_p[7501:10000], admix_mig$D_p[7501:10000]))
  
  # Test for bad values and then randomize the order
  if(sum(is.na(test$D)) > 0){
    test <- test[!is.na(test$D),]
  }
  test <- test[sample(nrow(test)),]
  
  if(test_plot){
    D_Dp <- rbind(train,test) %>% ggplot(aes(D,Dp, color=model)) +
      geom_point(alpha=0.5) +
      theme_bw() +
      labs(y=expression(D[p])) +
      ggtitle(expression(paste("Model space in D vs. ", D[p])))
    
    D_fhom <- rbind(train,test) %>% ggplot(aes(D,fhom, color=model)) +
      geom_point(alpha=0.5) +
      theme_bw() +
      labs(y=expression(f[hom])) +
      ggtitle(expression(paste("Model space in D vs. ", f[hom])))
    
    fhom_Dp <- rbind(train,test) %>% ggplot(aes(fhom,Dp, color=model)) +
      geom_point(alpha=0.5) +
      theme_bw() +
      labs(x=expression(f[hom]),
           y=expression(D[p])) +
      ggtitle(expression(paste("Model space in ", f[hom]," vs. ", D[p])))
    
    D_Dp + D_fhom + fhom_Dp
    ggsave(paste0("admix-stats-",cu,".pdf"), width = 14, height = 10)
  }
  
  rf <- abcrf(model ~ ., data=train, ntree = 1000)
  print(rf)
  
  rf_pred <- predict(rf, test, train)
  cfm <- confusionMatrix(rf_pred$allocation, test$model)
  print(cfm)
}
