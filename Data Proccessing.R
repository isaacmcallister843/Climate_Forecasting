path_data = "raw data path"
path_out = "output path"
library("tidyverse")

file.names <- dir(path_data, pattern ="csv")


proccess <- function(test_mod){
  date1 = str_split(test_mod[1, ][1], "-")[[1]][1]
  tav = 0 
  count = 0 
  test_final <- c()
  for (i in 1:nrow(test_mod)){
    date_ = test_mod[,1][i]
    date = str_split(test_mod[,1][i], "-")[[1]][1]
    if(date == date1){
      tav = tav + as.numeric(test_mod[,2][i])
      count = count + 1
    }
    else{
      tav = tav / count
      test_final = rbind(test_final, c(date_, tav))
      tav = 0 
      count = 0
      date1 = date
    }
    
  }
  test_final <- as.data.frame(test_final)
  rownames(test_final) <- 1:nrow(test_final)
  return(test_final) 
}

for(i in 1:length(file.names)){
  setwd(path_data)
  
  file <- read.csv(file.names[i])
  out_file <- cbind(file$Date, file$Temperature)
  out_file <- as.data.frame(out_file)
  
  for (item in 1:(nrow(out_file))){
    out_file[item,2] <- as.numeric(str_split(out_file[item,2],"Â")[[1]][1])
  }
  
  setwd(path_out)
  name = paste("Temperature", str_split(file.names[i], pattern = "_")[[1]][2])
  out_file <- proccess(out_file)
  print(nrow(out_file))
  write.csv(out_file,name)
}
