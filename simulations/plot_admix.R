library(tidyverse)

df_05 <- read_csv("admixture_0.5_tests.csv")
df_10 <- read_csv("admixture_1.0_tests.csv")
df_20 <- read_csv("admixture_2.0_tests.csv")
df <- rbind(df_05,df_10,df_20)

df$best_model <- factor(df$best_model,
                        levels=c("no_hyb",
                                 "hyb_sp",
                                 "admix",
                                 "admix_mig"))

df_correct <- df[df$best_model == "admix",]
df_incorrect <- df[df$best_model != "admix",]

cat("Mean weight for correct model: ", mean(df_correct$true_model_weight), "\n")
cat("Mean weight for incorrect model: ", mean(df_incorrect$best_model_weight), "\n")

ggplot(df, aes(x=gamma, y=admix_time)) +
  geom_point(aes(color=best_model_weight), size=0.2) +
  facet_grid(factor(time_units/2000.0)~best_model) +
  scale_color_gradient(low="grey90", high="grey10") +
  theme_bw(base_size = 8) +
  theme(axis.text.x = element_text(angle = 45, hjust=1)) +
  xlab("f") +
  ylab(expression(T[mix])) +
  ggtitle("Model Selection for Admixture Simulations")

ggsave("fig4.eps", width=169, height=150, units = "mm")
