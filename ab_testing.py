import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu

pd.set_option("max_columns", None)

######################
# UYGULAMA 1 : Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İsta. Ol. Anlamlı bir Fark var mıdır?
######################

df = sns.load_dataset("titanic")
df.groupby("sex").agg({"age": "mean"})  # Buradan bakıldığında erkeklerin yaş ortalaması fazla olarak gözüküyor.

# 1 - Hipotez Kur
# H0: M1  = M2 (Yani Kadın ve Erkek yaş ortalamaları arasında istatistiki olarak anlamlı bir fark yoktur.)
# H1: M1 != M2 (... vardır)

# 2 - Varsayım Kontrolü
# Normallik Varsayımı:
# HO: Normallik varsayımı sağlanmaktadır.
# H1: ... sağlanmamaktadır.

# 1.Grup İçin:
test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print("1. Grup Test Stat = %.4f, 1. Grup p-value = %.4f" % (test_stat, pvalue))
# p-value = 0.0071 < 0.05 oldugundan H0 Red edilir. Yani normallik varsayımı sağlanmaz.

# 2.Grup için:
test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print("2. Grup Test Stat = %.4f, 2. Grup p-value = %.4f" % (test_stat, pvalue))
# p-value = 0.0000 < 0.05 oldugundan H0 Red edilir. Yani normallik varsayımı sağlanmaz.

# Normallik varsayımını sağlamadığı için non-parametrik test olan mannwhitneyu testi uygulanır.

# Mannwhitneyu Testi:
test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value < 0.05 oldugundan H0 Reddedilir. Yani Kadın ve Erkek yaş ortalamaları arasında
# istatistiki olarak anlamlı bir fark vardır.


######################
# UYGULAMA 2: Diyabet hastası olan ve olmayanların yaşları arasında istatistiki olarak anl. bir fark var mıdır ?
######################

df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.info()
df.shape

# 1 - Hipotez testini kur.
# H0: M1  = M2 (Diyabet hastası olan ve olmayanların yaşları arasında istatistiki olarak anlamlı bir fark yoktur.)
# H1: M1 != M2 (... vardır)

# 2- Varsayımları kontrol et.

# Normallik Varsayımı (shapiro)
# H0: Normallik varsayımı sağlanır.
# H1: Sağlanmaz.

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value < 0.05 old. H0 reddedilir. Yani normallik varsayımı sağlanmaz.

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value < 0.05 old. H0 reddedilir. Yani normallik varsayımı sağlanmaz.

# Varyans Homojenliği Varsayımı (levene)
# H0: Varyans homojenliği vardır.
# H1: Yoktur.

test_stat, pvalue = levene(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                           df.loc[df["Outcome"] == 0, "Age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value > 0.05 old. H0 reddedilemez. Yani varyans homojenliği vardır.

# Normallik varsayımı sağlanmadı. Varyans homojenliği sağlandı. Non-parametrik olan mannwhitneyu testi yapılır.

# Mannwhitneyu Testi:
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value < 0.05 old. H0 reddedilir. Yani, Diyabet hastası olan ve olmayanların yaşları arasında
# istatistiki olarak anlamlı bir fark vardır.



