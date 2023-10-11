#install.packages("ggplot2")
#install.packages("psych")
#install.packages("reshape2")

#library(ggplot2)
#library(psych)
#library(reshape2)

# Set working directory
fpath <- "~/Documents/LTTS/"
setwd(fpath)

ann <- c('relu2d', 'sig2d', 'tanh2d', 'rbf2d', 'lstm2dval', 'gru2dval', 'trans2d', 'kgate2d')
n <- length(ann)

model <- c('nasdaq0704_', 'dj0704_', 'nikkei0704_', 'dax0704_')
m <- length(model)

data <- c('nasdaq_1_3_05-1_28_22.csv', 'dj_1_3_05-1_28_22.csv', 'nikkei_1_4_05_1_31_22.csv', 'dax_1_3_05_1_31_22.csv')
l <- length(data)

i <- 1
j <- 1

for (i in 1:n){
  for (j in 1:m){
    
    in_name1 <- paste('ws_att_err_', model[j], data[j],  '.', ann[i], '.0', '.txt', sep='')
    in_test1 <- read.delim(in_name1, sep = "", header = T, na.strings = " ", fill = T)

    in_name2 <- paste('ws_att_err_', model[j], data[j],  '.', ann[i], '.1', '.txt', sep='')
    in_test2 <- read.delim(in_name2, sep = "", header = T, na.strings = " ", fill = T)
    
        xh <- unlist(in_test1[, 2])
        yh <- unlist(in_test2[, 2])
    
        x <- unlist(in_test1[in_test1[,2] != in_test2[,2], 2])
        y <- unlist(in_test2[in_test1[,2] != in_test2[,2], 2])
        
        testg <- wilcox.test(x, y, paired = TRUE, alternative = "greater") #, mu=-0.02)
        testl <- wilcox.test(x, y, paired = TRUE, alternative = "less") #, mu=-0.02)
        test <- wilcox.test(x, y, paired = TRUE) #, mu=-0.02)
        
        xrh <- unlist(in_test1[, 3])
        yrh <- unlist(in_test2[, 3])
        
        xr <- unlist(in_test1[in_test1[,3] != in_test2[,3], 3])
        yr <- unlist(in_test2[in_test1[,3] != in_test2[,3], 3])
        
        testgr <- wilcox.test(xr, yr, paired = TRUE, alternative = "greater", mu=0) #-0.02)
        testlr <- wilcox.test(xr, yr, paired = TRUE, alternative = "less", mu=0) #-0.02)
        testr <- wilcox.test(xr, yr, paired = TRUE, mu=0) #-0.02)
        
        st <- sprintf( fmt="MAPE %s %s %s a0=%f+-%f a1=%f+-%f n/m=%d/%d %f/%f/%f", ann[i], model[j], data[j], mean(xh), sd(xh), mean(yh), sd(yh), length(xh), length(x), testg$p.value, testl$p.value, test$p.value)
        print(st)
        st <- sprintf( fmt="RMSE %s %s %s a0=%f+-%f a1=%f+-%f n/m=%d/%d %f/%f/%f", ann[i], model[j], data[j], mean(xrh), sd(xrh), mean(yrh), sd(yrh),  length(xrh), length(xr), testgr$p.value, testlr$p.value, testr$p.value)
        print(st)
        
        flush.console()

  }
}