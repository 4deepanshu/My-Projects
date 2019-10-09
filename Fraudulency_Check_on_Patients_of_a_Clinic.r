
file <- read.csv("D:/Project Discussion_Workng on Case study/Project Discussion_Workng on Case study/Health_claim.csv")
summary(file)
View(file)
plot(file)
#barplot(file$Distance_from_clinic,file$Claim_amount,type='b',main="GRAPH",col.main="Red",xlab="Index",ylab="Values",col.lab="Green",col=rgb(0,0.43,1),font.main=7)
boxplot(file,type='b',main="GRAPH",col.main="Red",xlab="Index",ylab="Values",col.lab="Green",col=rgb(0,0.43,1),font.main=7)
plot(file$age,file$Claim_amount,col.main="Red",xlab="Age",ylab = "Claim Amount")
plot(file$Distance_from_clinic,file$Claim_amount)
barplot(file$age,file$Num_medical_bills,col="red")

file <- na.omit(file)
rang = nrow(file)
#file <- readLines("D:/Project Discussion_Workng on Case study/Project Discussion_Workng on Case study/Health_claim.csv")
#file <- scan("D:/Project Discussion_Workng on Case study/Project Discussion_Workng on Case study/Health_claim.csv",skip=2)
rang

nrow(file)
View(file)
yong <-rep(0,each=36258)
yong

# Marking Distance_from_clinic = 0 as Fradulant
j <- rep(0,each=50000)
j
k=0
d=0
u <- file[,3]
u
for(i in u)
{
  k=k+1
  if(i==0)
  {
    d=d+1
    j[d]=k
    yong[k] <- "Fradulant" 
  }
  else
  {
    yong[k] <- "Not Fradulant"
    next
  }
}
j
yong # Fradulant or not Stored in the form of vector
sam <- file[-j,]  # Omiting Fradualnt Storind people's data who are not fradulant
sam

# Marking no. of medical bills = 0 as fradulant 
qq <- rep(0,each=50000)
qq
tt=0
ee=0
bu <- sam[,5]
bu
for(i in bu)
{
  tt=tt+1
  if(i==0)
  {
    ee=ee+1
    qq[ee]=tt
    yong[tt] <- "Fradulant"
  }
  else
  {
    next
  }
}
qq

sam1 <- sam[-qq,]
sam1    # Omitted fradulant


# Marking claim amount = 0 as fradulant

rt <- rep(0,each=50000)
rt
ty=0
ff=0
lu <- sam1[,6]
lu
for(i in lu)
{
  ty=ty+1
  if(i==0)
  {
    ff=ff+1
    rt[ff]=ty
    yong[ty] <- "Fradulant" 
  }
  else
  {
    next
  }
}
rt
yong
sam2 <- sam1[-rt,]  # Omitted fradulant cases
sam2

barplot(sam2$age,sam2$Num_medical_bills,col="red")
nrow(sam2)
nrow(file)
file <- cbind(file,yong)
file
View(file)

# outlier detection
boxplot(file$Distance_from_clinic,ylab="Distance")
max(file$Distance_from_clinic)
min(file$Distance_from_clinic)
median(file$Distance_from_clinic)
summary(file$Distance_from_clinic)
quart1= summary(file$Distance_from_clinic)[2]
quart3= summary(file$Distance_from_clinic)[5]
IQR = quart3-quart1
IQR
outlier=quart3+IQR*1.5
outlier


# Outlier detection and detection  of cases prone to be fradualant

pp <- rep(0,each=5000)
sw=0
ti=0
ua <- file[,3]
ua
for(i in ua)
{
  sw=sw+1
  if(i > outlier)
  {
    ti=ti+1
    pp[ti]=sw
    yong[sw] <- "Fradulant" 
  }
}
yong
sam3 <- sam2[-pp,]
sam3


ce <- rep(0,each=5000)
pw=0
tir=0
ual <- file[,6]
ual
for(i in ual)
{
  pw=pw+1
  if(i > outlier)
  {
    tir=tir+1
    ce[tir]=sw
    yong[pw] <- "Fradulant" 
  }
}
sam4 <- sam3[-ce,]
sam4       # Ommitted fradulant cases
file

# Total Fraud
total_claim_amount=sum(file$Claim_amount)
total_claim_amount
tapply(file$Claim_amount,yong,sum)

# The cases prone to be fradulant
file <- cbind(file,yong)
tapply(file$Claim_amount , yong, sum)
file <- file[,1:9]
View(file)

#  K means

tw=numeric(15)
for( i in 1:15 )
{
  tw[i] = kmeans(file$Distance_from_clinic,i)$tot.withinss
}
plot(tw,type = "l",col="green",main="Determining value in K means",col.main="blue")
# from this k =6

clustering <- kmeans(file$Distance_from_clinic,centers = 6)
clustering

final <- file
file


# Training data
library(caTools)

split = sample.split(file, SplitRatio = 0.9)
split

train <- subset(file ,split == TRUE)
test <- subset(file , split == FALSE)

model <- lm(yong~.,data=train)
pred <- predict(model,test)
pred
plot(pred,main="Predicted values from multiple regression")

confmat = table(test$yong,pred)
confmat
accuracy = sum(diag(confmat))/nrow(test)
accuracy


# Decision Trees

library(rpart)
library(rpart.plot)
library(caTools)
model <- rpart(file$yong~.,data=file,method="class")
model
rpart.plot(model,type=3,extra=101)

pred <- predict(model,file, type ="class")
pred
file
confmat = table(file$yong,pred)
confmat
accuracy = sum(diag(confmat))/nrow(test)
accuracy

#Time series 
library(plyr)
library(caTools)
library(forecast)

attach(file)

mydata<- file[order(Patient_id),]
mydata

mydatats <- ts(mydata$Claim_amount ,frequency = 12)
mydatats
summary(mydatats)

plot(mydatats,col="blue")
decomp <- decompose(mydatats)
plot(decomp,col="red")
set.seed(100)

split = sample.split(mydatats, SplitRatio = 0.9)
split

train <- subset(mydatats ,split == TRUE)
test <- subset(mydatats , split == FALSE)
nrow(test)
nrow(train)
train
test

dtrain <- diff(train ,differences = 2)
plot(dtrain,col="green",main="ARIMA Differencing")

modell <- auto.arima(dtrain)
modell 
plot(modell,main="Arima model",col="red")
myforecast <- forecast(object = modell, test)
myforecast
plot(myforecast,col="green")

detach(file)

# Use ggplot to visualize the general public whos not fradulant
library(ggplot2)

file1 <- cbind(Age=file$age,Claim_amount=file$Claim_amount,file$yong) 
file1
file1 <- as.data.frame(file1)
View(file1)

file
p <- ggplot(data=file,aes(age,Claim_amount))
p + geom_area(aes(colour = factor(Num_medical_bills))) + ggtitle("Age Vs Claim_amount GRAPH",colors(distinct = FALSE))

file1
p <- ggplot(data=file1,aes(Age,Claim_amount))
p + geom_area(aes(colour = factor(yong))) + ggtitle("Age Vs Claim_amount GRAPH")

# dataset having no frauds
datafile <- sam4
datafile
ggplot(data = datafile,aes(age,Claim_amount)) + geom_point(aes(colour = factor(Num_medical_bills))) + ggtitle("Age Vs Claim amount GRAPH")
ggplot(data = datafile,aes(age,Claim_amount)) + geom_point(aes(colour = factor(datafile$Days_admitted))) + ggtitle("Age Vs Claim amount GRAPH")
ggplot(data = datafile,aes(age,Claim_amount)) + geom_point(aes(colour = factor(datafile$Month))) + ggtitle("Age Vs Claim amount GRAPH") 
ggplot(data = datafile,aes(age,Claim_amount)) + geom_point(aes(colour = factor(datafile$Month))) + ggtitle("Age Vs Claim amount GRAPH") + scale_shape(solid = FALSE)

class(datafile)

# Genuine data
datafile
mean(datafile$Distance_from_clinic)
median(datafile$Distance_from_clinic)

mean(datafile$age)
median(datafile$age)

mean(datafile$Distance_from_clinic)
median(datafile$Distance_from_clinic)

mean(datafile$Days_admitted)
median(datafile$Days_admitted)

mean(datafile$Num_medical_bills)
median(datafile$Num_medical_bills)

mean(datafile$Claim_amount)
median(datafile$Claim_amount)

median(datafile$Month)
