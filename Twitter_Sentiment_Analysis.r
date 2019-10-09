library(twitteR)
library(wordcloud)
library(tm)
library(ggplot2)
library(gridExtra)
library(plyr)
library(RColorBrewer)
library(sentimentr)
library(plyr)
library(stringr)
library(dplyr)

api_key <-  "********************"
api_secret <-  "************************"
token <-"*************************************"
token_secret <- "************************"

setup_twitter_oauth(api_key,api_secret,token,token_secret)

tweet <- searchTwitter("from:@realDonaldTrump",n=2000,lang="en")
#tweet <- searchTwitter("from:@@NarendraModi",n=200,lang="en")
tweet

df <- twListToDF(tweet)
df

stack <- df$text
stack

setwd("D:/Time-Series Analysis in R - 17th Feb/Time-Series Analysis in R - 17th Feb/forstudents - 17th & 18th Feb Sessions/forstudents-day9and10/")
pos <- readLines("positive.txt")
neg <- readLines("negative.txt")
pos
pos.words <- c(pos,"ugrade")
neg.words <- c(neg,"wait","bla","Epicfail")

score.sentiment = function(sentences, pos.words, neg.words, .progress='none')
  {
  scores = laply(sentences,function(sentence, pos.words, neg.words)
                   {
                   
                   sentence = gsub("[[:punct:]]", "", sentence)
                   sentence = gsub("[[:cntrl:]]", "", sentence)
                   sentence = gsub('\\d+', '', sentence)
                   
                   tryTolower = function(x)
                     {
                     y = NA
                     try_error = tryCatch(tolower(x), error=function(e) e)
                     if (!inherits(try_error, "error"))
                       y = tolower(x)
                     return(y)
                     }
                   
                   sentence = sapply(sentence, tryTolower)
                   word.list = str_split(sentence, "\\s+")
                   words = unlist(word.list)
                   pos.matches = match(words, pos.words)
                   neg.matches = match(words, neg.words)
                   pos.matches = !is.na(pos.matches)
                   neg.matches = !is.na(neg.matches)
                   
                   score = sum(pos.matches)-sum(neg.matches)
                   return(score)
                   }, pos.words, neg.words, .progress=.progress )
  
  scores.df = data.frame(text=sentences, score=scores)
  return(scores.df)
  }

Dataset <- stack 
Dataset

scores <- score.sentiment(Dataset, pos.words, neg.words, .progress='text')

stat <- scores
stat <- mutate(stat, tweet=ifelse(stat$score > 0, 'positive', ifelse(stat$score < 0, 'negative', 'neutral')))
View(stat)


# Machine learning

# machine learning cannot be applied , makes no sense as the words in senntences are also taken as individual blba bla akes no sense


write.csv(stat,file="sentiment_results.csv", row.names = F)

sum(table(stat$tweet))
lbls <- c('negative','neutral','positive')
pct <- round(table(stat$tweet)/sum(table(stat$tweet))*100)
lbls <- paste(lbls, pct) 
lbls <- paste(lbls,"%",sep="")  

pie(table(stat$tweet), labels = lbls, col=rainbow(3))

corpus <- Corpus(VectorSource(Dataset))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus,removeNumbers)
corpus <- tm_map(corpus,removePunctuation)
corpus <- tm_map(corpus,removeWords, c(stopwords("english")))
dtm <- TermDocumentMatrix(corpus) 
m <- as.matrix(dtm)
wordfreq <- rowSums(m)
d <- data.frame(word = names(wordfreq),freq=wordfreq)


require(wordcloud)
set.seed(1234)
wordcloud(words= d$word, freq =d$freq, min.freq = 5000,max.words = Inf,colors=brewer.pal(n=8,name="Dark2"))



