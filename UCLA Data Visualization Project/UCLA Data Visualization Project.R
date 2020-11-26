#################################################
## Exploratory Data Analysis and Visualization
## Name: Tianpeng Gai (Leo)
## Last Modified: Mar 1st, 2020
#################################################

# Import library
library(readr)
library(ggplot2)
library(corrplot)
library(GGally)
library(car)
library(rgl)
library(RColorBrewer)
library(cluster)

# Read in raw dataset
FGPdat <- read_csv("TOP 500 YouTubers.csv")

# Preparation
options(scipen = 999)
colnames(FGPdat) <- c("Rank", "Subscribers", "Grades", "YouTuber", "Uploads", "Views")

# Separate FGPdat into two datasets based on NAs 
# (The rows with NAs are the YouTube subsites based on different YouTube topics, e.g. YouTube Gaming, YouTube Music)
# (Or the YouTube channel that was closed but still with millions of Subscribers, e.g. Machinima was closed on Jan 18th, 2019)
FGPdat_cat <- FGPdat[is.na(FGPdat$Views),]
FGPdat_main <- FGPdat[!is.na(FGPdat$Views),]

# Remove the columns contain NAs, the row contains "Machinima" and column "Rank" in FGPdat_cat 
# (Machinima was closed for a while, suggest dropping it from the dataset)
# (The column "Rank" is based on subscribers which is redundant)
FGPdat_cat$Grades <- NULL
FGPdat_cat$Uploads <- NULL
FGPdat_cat$Views <- NULL
FGPdat_cat$Rank <- NULL
FGPdat_cat <- FGPdat_cat[!grepl("Machinima", FGPdat_cat$YouTuber),]

# Change the column "Grades" to factorial variable, remove the column "Rank" in FGPdat_main 
# (The column "Rank" is based on subscribers which is redundant)
FGPdat_main$Rank <- NULL
FGPdat_main$Grades <- factor(FGPdat_main$Grades, 
                             levels = c("A++","A+","A","A-","B+","B","B-","C+","D-"), 
                             labels = c("A++","A+","A","A-","B+","B","B-","C+","D-"))

# Checking datasets
head(FGPdat_cat); summary(FGPdat_cat); table(FGPdat_cat$YouTuber)
head(FGPdat_main); summary(FGPdat_main); table(FGPdat_main$YouTuber)

# Export new datasets 
write_excel_csv(FGPdat_cat, "FGPdat_cat.csv")
write_excel_csv(FGPdat_main, "FGPdat_main.csv")

# Summary statistics (FGPdat_cat) - Subscribers boxplot with outliers
ggplot(FGPdat_cat, aes(x = "", y = Subscribers)) + 
  geom_boxplot(fill = "gray") +
  labs(title = "Summary statistics (FGPdat_cat) - Subscribers boxplot with outliers", x = "", y = "Subscribers in millions") + 
  theme_gray(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
        axis.title.y = element_text(face = "bold", color = "black", size = 14), 
        axis.title.x = element_text(face = "bold", color = "black", size = 14),
        axis.text.y = element_text(face = "bold", color = "black", size = 10),
        axis.text.x = element_text(face = "bold", color = "black", size = 10))

# Summary statistics (FGPdat_main) - Subscribers boxplot with outliers
ggplot(FGPdat_main, aes(x = "", y = Subscribers)) + 
  geom_boxplot(fill = "gray") +
  labs(title = "Summary statistics (FGPdat_main) - Subscribers boxplot with outliers", x = "", y = "Subscribers in millions") + 
  theme_gray(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
        axis.title.y = element_text(face = "bold", color = "black", size = 14), 
        axis.title.x = element_text(face = "bold", color = "black", size = 14),
        axis.text.y = element_text(face = "bold", color = "black", size = 10),
        axis.text.x = element_text(face = "bold", color = "black", size = 10))

# Summary statistics (FGPdat_main) - Uploads boxplot with outliers
ggplot(FGPdat_main, aes(x = "", y = Uploads)) + 
  geom_boxplot(fill = "gray") +
  labs(title = "Summary statistics (FGPdat_main) - Uploads boxplot with outliers", x = "", y = "Uploads") + 
  theme_gray(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
        axis.title.y = element_text(face = "bold", color = "black", size = 14), 
        axis.title.x = element_text(face = "bold", color = "black", size = 14),
        axis.text.y = element_text(face = "bold", color = "black", size = 10),
        axis.text.x = element_text(face = "bold", color = "black", size = 10))

# Summary statistics (FGPdat_main) - Views boxplot with outliers
ggplot(FGPdat_main, aes(x = "", y = Views)) + 
  geom_boxplot(fill = "gray") +
  labs(title = "Summary statistics (FGPdat_main) - Views boxplot with outliers", x = "", y = "Views") + 
  theme_gray(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
        axis.title.y = element_text(face = "bold", color = "black", size = 14), 
        axis.title.x = element_text(face = "bold", color = "black", size = 14),
        axis.text.y = element_text(face = "bold", color = "black", size = 10),
        axis.text.x = element_text(face = "bold", color = "black", size = 10))

# YouTube topics by subscribers bar chart
ggplot(FGPdat_cat, aes(x = reorder(YouTuber, -Subscribers), y = Subscribers)) + 
  geom_bar(stat = 'identity') + 
  labs(title = "YouTube topics by subscribers bar chart", x = "YouTube topics", y = "Subscribers in millions") + 
  theme_gray(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
        axis.title.y = element_text(face = "bold", color = "black", size = 14), 
        axis.title.x = element_text(face = "bold", color = "black", size = 14),
        axis.text.y = element_text(face = "bold", color = "black", size = 10),
        axis.text.x = element_text(face = "bold", color = "black", size = 10))

# Top 10 YouTube channels by subscribers bar chart
Top_10 <- FGPdat_main[1:10,]
ggplot(Top_10, aes(x = reorder(YouTuber, -Subscribers), y = Subscribers)) + 
  geom_bar(stat = 'identity') + 
  labs(title = "Top 10 YouTube channels by subscribers bar chart", x = "YouTube channels", y = "Subscribers in millions") + 
  theme_gray(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
        axis.title.y = element_text(face = "bold", color = "black", size = 14), 
        axis.title.x = element_text(face = "bold", color = "black", size = 14),
        axis.text.y = element_text(face = "bold", color = "black", size = 9),
        axis.text.x = element_text(face = "bold", color = "black", size = 9))

# Subscribers by grades density plot
ggplot(FGPdat_main, aes(x = Subscribers, fill = Grades)) +
  geom_density(alpha = 0.5) + 
  labs(x = "Subscribers in millions", y = "Density", title = "Subscribers by grades distribution", fill = "Grades") + 
  theme_gray(base_size = 14)
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
        axis.title.y = element_text(face = "bold", color = "black", size = 14), 
        axis.title.x = element_text(face = "bold", color = "black", size = 14),
        axis.text.y = element_text(face = "bold", color = "black", size = 12),
        axis.text.x = element_text(face = "bold", color = "black", size = 12),
        legend.title = element_text(size = 10))

# Coefficients among Subscribers, Uploads and Views
ggpairs(data = FGPdat_main, columns = c(1,4:5), title = "Strength of correlations between variables")
cormat <- cor(FGPdat_main[,c(1,4:5)])
corrplot(cormat, method = "color")
corrplot(cormat, method = "number")

# Remove the most absurd outliers
FGPdat_main <- FGPdat_main[-which(FGPdat_main$Views>42000000000),]

# Formatting Views in billions
FGPdat_main$Views <- FGPdat_main$Views/1000000000

# The boxplot after removing views greater than forty-two billion
ggplot(FGPdat_main, aes(x = "", y = Views)) + 
  geom_boxplot(fill = "gray") +
  labs(title = "Removed views greater than forty-two billion", x = "", y = "Views in billions") + 
  theme_gray(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
        axis.title.y = element_text(face = "bold", color = "black", size = 14), 
        axis.title.x = element_text(face = "bold", color = "black", size = 14),
        axis.text.y = element_text(face = "bold", color = "black", size = 10),
        axis.text.x = element_text(face = "bold", color = "black", size = 10))

# Uploads, Views, Subscribers by grades scatter3d plot
colors <- brewer.pal(9, "Set1")
scatter3d(x = FGPdat_main$Uploads, 
          y = FGPdat_main$Views, 
          z = FGPdat_main$Subscribers, 
          xlab = "Uploads", 
          ylab = "Views in billions",
          zlab = "Subscribers in millions",
          groups = FGPdat_main$Grades,
          grid = FALSE, 
          fit = "smooth",
          surface.col = colors,
          revolutions = 2)

# Uploads by Subscribers clustering distribution
k <- 5         
kmeans_5 <- kmeans(FGPdat_main[,c("Uploads","Subscribers")], k)
plot(FGPdat_main[,c("Uploads","Subscribers")], 
     col = kmeans_5$cluster, 
     main = "Uploads by Subscribers clustering distribution",
     xlab = "Uploads",
     ylab = "Subscribers in millions")
points(kmeans_5$centers, col = 1:k, pch = 8, cex = 2)


# Try to build a regression model
model <- lm(Subscribers~Views, data = FGPdat_main)
summary(model)
plot(model)

# Views by Subscribers regression model
ggplot(FGPdat_main, aes(x = Views, y = Subscribers)) + 
  geom_point(shape = 16, size = 1) + 
  geom_smooth(method = "lm", level = 0.90) + 
  labs(title = "Views by Subscribers regression model", x = "Views in billions", y = "Subscribers in millions") + 
  scale_x_continuous(breaks = seq(0, 42, 10)) + 
  theme_gray(base_size = 14) + 
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
        axis.title.y = element_text(face = "bold", color = "black", size = 14), 
        axis.title.x = element_text(face = "bold", color = "black", size = 14),
        axis.text.y = element_text(face = "bold", color = "black", size = 10),
        axis.text.x = element_text(face = "bold", color = "black", size = 10))

