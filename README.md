# Mean-Variance-Hedging

This repository is an exploration of mean-variance hedging in discrete time, in particular the Deep Hedging strategy from **Hans Buehler, Lukas Gonon, Josef Teichmann, and Ben Wood.Deep Hedging. 2018** and the dynamic programming hedging

Jupyter Notebooks are available in the `examples/` folder, and also accessible online in Google Colab (listed below)

To run the examples locally, ensure that all dependencies are installed by running `pip install -r requirements.txt`, and ensure that the `mean_variance_hedge` folder is used as a directory for the corresponding utilities and functions to be used.

Documentation on how to use the functions within `mean_variance_hedge` is available at : [https://chrischia06.github.io/mean-Variance-Hedging/mean_variance_hedge/](chrischia06.github.io/mean-Variance-Hedging/mean_variance_hedge/)

## Notebooks

**Dynamic Programming Hedge**

+ [Colab Notebook](https://colab.research.google.com/drive/1yptQ4xmVzUhSarbbfh5Y80FR4h7MM-Ki)

**Deep Hedging in a Multinomial Tree setting**

+ `examples/DeepHedgingInLattice.ipynb`
+ Accessible online as a [Colab Notebook](https://colab.research.google.com/drive/1Zzc0BUcjt9Pxa7Z-5JiGx5GMklAg9WZB?usp=sharing)

**Deep Hedging in the Black Scholes setting**

+ `examples/DeepHedgingBlackScholes.ipynb`
+ Accessible online as a [Colab Notebook](https://colab.research.google.com/drive/1Zzc0BUcjt9Pxa7Z-5JiGx5GMklAg9WZB?usp=sharing)

**Deep Hedging in a Heston setting**

+ `examples/DeepHedgingHeston.ipynb`
+ Accessible online as a [Colab Notebook](https://colab.research.google.com/drive/1Zzc0BUcjt9Pxa7Z-5JiGx5GMklAg9WZB?usp=sharing)

Deep Hedging in a Rough Bergomi setting

+ `examples/DeepHedgingRoughBergomi.ipynb`
+ Accessible online as a [Colab Notebook](https://colab.research.google.com/drive/1Zzc0BUcjt9Pxa7Z-5JiGx5GMklAg9WZB?usp=sharing)


# References

**Černý, Aleš, Dynamic Programming and Mean-Variance Hedging in Discrete Time (October 1, 2003). Applied Mathematical Finance, 2004, 11(1), 1-25, Available at SSRN: https://ssrn.com/abstract=561223**

**Hans Buehler, Lukas Gonon, Josef Teichmann, and Ben Wood.Deep Hedging. 2018. arXiv:1802.03042 [q-fin.CP].**

**Johannes  Ruf  and  Weiguan  Wang.Hedging  with  Neural  Networks.  2020.arXiv:2004.08891 [q-fin.RM]**

**Cornelis  W  Oosterlee  and  Lech  A  Grzelak.Mathematical  Modelling  and Computation in Finance. World Scientific (Europe), 2019.**

+ And code from: [https://github.com/LechGrzelak/Computational-Finance-Course](https://github.com/LechGrzelak/Computational-Finance-Course)

**Ryan  McCrickerd  and  Mikko  S  Pakkanen.  “Turbocharging  Monte  Carlopricing for the rough Bergomi model”. In:Quantitative Finance18.11 (2018),pp. 1877–1886.**

+ And code from: [https://github.com/ryanmccrickerd/rough_bergomi](https://github.com/ryanmccrickerd/rough_bergomi)

**Blanka  Horvath,  Josef  Teichmann,  and  Zan  Zuric.Deep  Hedging  underRough Volatility. 2021. arXiv:2102.01962 [q-fin.CP]**

