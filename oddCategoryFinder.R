R Code:

library(sqldf)
MyData <- read.csv('train.csv', header=FALSE, sep=",")
Data <- unique(MyData)
names(Data) <- c('product_title', 'brand_id','category_id')
head(Data)

DF=sqldf("select category_id,brand_id,count(brand_id) as z from Data group by category_id,brand_id ")
DF1 = sqldf("SELECT category_id,brand_id FROM   DF group by 1 having max(z)");
DF1