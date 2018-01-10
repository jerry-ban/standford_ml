########################################################################
#                     R Script for Pivoting Data                       #
#               Written by Mike Piserchia - 10/4/2017                  #
#    (Please do not hesitate to Skype or email me with questions)      #
########################################################################

########################################################################
# Steps to pivot your dataset:                                         #
# 1. Highlight and run this entire script (Ctrl+A+Enter is pretty      #
#    quick).                                                           #
# 2. Follow the pop-up window prompts until the pivot table opens on   #
#    your desktop.                                                     #
#                                                                      #
# NB: Pivot tables will also be saved to your desktop.                 # 
########################################################################
####################################################################################################################################
answer<-winDialog("yesno", "Is this the first pivot table you are creating this session?")
if (answer=='YES') {

required_packages <- c("data.table", "svDialogs", "plyr", "dplyr")

for(i in 1:length(required_packages)){
  if(required_packages[i] %in% installed.packages()[,1]) 
    {library(required_packages[i], character.only=T)}
    else{install.packages(required_packages[1])
         library(required_packages[i], character.only=T)}  
}
####################################################################################################################################
data_path <- dlgOpen(title = "Please select a csv file", filters = dlgFilters[c("R", "All"), ])$res
raw_data_reference <- read.csv(data_path, header=T, stringsAsFactors=FALSE)

variables <- names(raw_data_reference)

numeric_vars <- dlgList(variables, title="Which variable(s) are numeric?", multiple=T)$res
numeric_cols <- which(names(raw_data_reference) == numeric_vars)
raw_data_reference[,numeric_cols] <- sapply(raw_data_reference[,numeric_cols],as.numeric)
}
####################################################################################################################################
data <- raw_data_reference
variables <- names(data)

filter_labels <- NULL
answer<-winDialog("yesno", "Do you need to filter on a variable(s)?")
if (answer=='YES') {
     filter_vars <- dlgList(variables, title="Filter on which variable(s)?", multiple=T)$res

      filter_levels <- list() ; filter_specs <- list() ; filter_labels <- NULL
      for(i in 1:length(filter_vars)){
        filter_levels[[i]] <- unique(data[,which(names(data)==filter_vars[i])])
        filter_specs[[i]] <- dlgList(filter_levels[[i]], title=paste("Which", filter_vars[i], "level(s)?"), multiple=T)$res
        
        filter_labels[i] <- paste(filter_vars[i],": ", paste(unlist(filter_specs[i]), collapse=", "))
        
        filter_cols <- which(names(data) == filter_vars[i])
  
        data <- data[data[,filter_cols] %in% unlist(filter_specs[i]),]
      }
}
####################################################################################################################################
rows <- dlgList(variables, title="Row Variable(s)?", multiple=T)$res
rows <- paste(rows, collapse=" + ")

col_answer<-winDialog("yesno", "Do you need a column dimension(s)?")
if (col_answer=='YES') {
columns <- dlgList(variables, title="Column Variable(s)?", multiple=T)$res
columns <- paste(columns, collapse=" + ")

pivot_specs <- paste(rows, "~", columns)
} else{pivot_specs <- paste(rows, "~ .")}

value <- dlgList(variables, title="Variable to pivot?", multiple=F)$res

pivot_types <- c("sum", "mean","count", "min", "max", "standard deviation", "variance")
pivot_type <- dlgList(pivot_types, title="Which form of aggregation?", multiple=F)$res
  if(pivot_type=="count"){pivot_type <- "length"}
  if(pivot_type=="standard deviation"){pivot_type <- "sd"}
  if(pivot_type=="variance"){pivot_type <- "var"}

pivot_table <- dcast.data.table(as.data.table(data), pivot_specs, value.var= value, fun.aggregate = match.fun(pivot_type), na.rm=T)
final_dataset <- pivot_table
####################################################################################################################################
if(!is.null(filter_labels)){                                                                                

  final_dataset <- as.data.frame(rbind(filter_labels,final_dataset, fill=T), stringAsFactors=F)                       
  final_dataset[is.na(final_dataset)] <- ""
  names(final_dataset)[1] <- "Filters" 

  new_names_row <- max(which(final_dataset[,1] != ""))
  final_dataset[new_names_row,-1] <- names(final_dataset[-1])
  names(final_dataset)[-1] <- ""  
  
  if(col_answer=='NO'){
    final_dataset[new_names_row,ncol(final_dataset)] <- value
    names(final_dataset)[-1] <- ""
  } 
}                                                                                                          
####################################################################################################################################
if(col_answer=='NO' &  is.null(filter_labels)== TRUE){
  names(final_dataset)[2] <- value
}
####################################################################################################################################
export_path <- paste0(file.path(Sys.getenv("USERPROFILE"),"Desktop"), "/",pivot_type, "-pivot- ", value, " by ", pivot_specs," ", 
                      gsub(":", ".", Sys.time()), ".csv")
write.csv(final_dataset, export_path, row.names=F)
shell.exec(export_path)
####################################################################################################################################